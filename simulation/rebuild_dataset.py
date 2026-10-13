"""Rebuild the normalized ski-jump dataset with validated timestamps.

Raw CSV files are never modified.  Each file's pre-flight record is removed
once, source row order is retained, and timestamp values outside the earliest
longest strictly increasing subsequence are reconstructed by linear
interpolation between trusted neighbours.  The repaired time axis is asserted
to be strictly increasing before 0.05 s resampling.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flatten_wind import flatten_wind_blob


SIMULATION_DIR = Path(__file__).resolve().parent
RAW_DIR = SIMULATION_DIR / "2024_03_Planica_12_winds"
OUTPUT_DIR = RAW_DIR / "cleaned" / "normalized_corrected"
STEP_SECONDS = 0.05
EXCLUDED_FILE = (
    "999_999_Jumper_Anon_ANO_1_20240409-103759_C_OfficialResults.csv"
)

ROLL_COLUMNS = ["Roll Angle Left [°]", "Roll Angle Right [°]"]
BODY_COLUMNS = [
    "Opening Angle [°]",
    *ROLL_COLUMNS,
    "Yaw Angle Left [°]",
    "Yaw Angle Right [°]",
    "Stalling Angle Left [°]",
    "Stalling Angle Right [°]",
]
ZONES = [
    ["W1", "W2", "W3", "W4"],
    ["W5", "W6", "W7", "W8"],
    ["W9", "W10", "W11", "W12"],
]
WIND_FEATURES = ["Speed", "Tangent", "Cross", "Turbulence"]


def parse_timestamps(values: pd.Series) -> np.ndarray:
    """Convert timezone-aware ISO timestamps to seconds from the first value."""
    parsed: list[datetime] = []
    for value in values.astype(str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Unparseable timestamp: {value!r}") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"Timestamp lacks a UTC offset: {value!r}")
        parsed.append(timestamp)

    first = parsed[0]
    return np.asarray([(timestamp - first).total_seconds() for timestamp in parsed])


def earliest_longest_increasing_indices(values: np.ndarray) -> np.ndarray:
    """Return an earliest longest strictly increasing subsequence.

    The quadratic dynamic program is inexpensive for these approximately
    200-row files.  Iterating predecessors in source order and retaining the
    first equal-length solution prevents a later, slightly smaller timestamp
    from displacing an earlier physically ordered record.
    """
    count = len(values)
    lengths = np.ones(count, dtype=int)
    predecessors = np.full(count, -1, dtype=int)
    for right in range(count):
        for left in range(right):
            candidate = lengths[left] + 1
            if values[left] < values[right] and candidate > lengths[right]:
                lengths[right] = candidate
                predecessors[right] = left

    maximum = int(lengths.max())
    if maximum < 2:
        raise ValueError("Cannot reconstruct timestamps from fewer than two trusted rows")
    endpoint = int(np.flatnonzero(lengths == maximum)[0])
    indices: list[int] = []
    while endpoint >= 0:
        indices.append(endpoint)
        endpoint = int(predecessors[endpoint])
    return np.asarray(indices[::-1], dtype=int)


def repair_timestamps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repair non-monotone timestamps without reordering measurement rows."""
    trusted = earliest_longest_increasing_indices(values)
    repaired = values.astype(float, copy=True)
    repaired_indices = np.setdiff1d(np.arange(len(values)), trusted)
    if len(repaired_indices) == 0:
        return repaired, repaired_indices

    repaired[:] = np.interp(np.arange(len(values)), trusted, values[trusted])
    trusted_steps = np.diff(values[trusted]) / np.diff(trusted)
    typical_step = float(np.median(trusted_steps[trusted_steps > 0]))
    if trusted[0] > 0:
        offsets = trusted[0] - np.arange(trusted[0])
        repaired[: trusted[0]] = values[trusted[0]] - typical_step * offsets
    if trusted[-1] < len(values) - 1:
        offsets = np.arange(trusted[-1] + 1, len(values)) - trusted[-1]
        repaired[trusted[-1] + 1 :] = values[trusted[-1]] + typical_step * offsets

    if np.any(np.diff(repaired) <= 0):
        raise ValueError("Timestamp repair did not produce a strictly increasing axis")
    return repaired, repaired_indices


def validate_and_remove_preflight(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    first = df.iloc[0]
    posture_missing = pd.isna(first[BODY_COLUMNS]).all()
    zero_speed = float(first["Speed resulting [km/h]"]) == 0.0
    negative_x = float(first["X [m]"]) < 0.0
    if not (posture_missing and zero_speed and negative_x):
        raise ValueError(f"{filename}: first row does not match the pre-flight record")
    return df.iloc[1:].reset_index(drop=True)


def flatten_raw_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";")
    wind_column = raw.columns[-1]
    wind = raw[wind_column].apply(flatten_wind_blob)
    flattened = pd.concat([raw.drop(columns=[wind_column]), wind], axis=1)
    flattened["Time"] = parse_timestamps(flattened["Time"])
    return validate_and_remove_preflight(flattened, path.name)


def resample(df: pd.DataFrame, repaired_time: np.ndarray) -> pd.DataFrame:
    new_time = np.arange(repaired_time[0], repaired_time[-1], STEP_SECONDS)
    output: dict[str, np.ndarray] = {"Time": new_time}
    for column in df.columns:
        if column == "Time" or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        output[column] = np.interp(
            new_time,
            repaired_time,
            df[column].to_numpy(dtype=float),
        )
    return pd.DataFrame(output)


def build_model_arrays(df: pd.DataFrame, filename: str):
    required_without_roll = [
        "X [m]",
        "Y [m]",
        "Z [m]",
        "Height above ground [m]",
        "Speed resulting [km/h]",
        *[column for column in BODY_COLUMNS if column not in ROLL_COLUMNS],
        *[
            f"{sensor}_{feature}"
            for feature in WIND_FEATURES
            for zone in ZONES
            for sensor in zone
        ],
    ]
    missing_other = int(df[required_without_roll].isna().sum().sum())
    if missing_other:
        raise ValueError(f"{filename}: {missing_other} unexpected non-roll missing cells")

    roll_missing = int(df[ROLL_COLUMNS].isna().sum().sum())
    df = df.copy()
    df[ROLL_COLUMNS] = df[ROLL_COLUMNS].ffill().bfill()
    if df[ROLL_COLUMNS].isna().any().any():
        raise ValueError(f"{filename}: roll-angle filling left missing values")

    x = df["X [m]"].to_numpy()
    y = df["Y [m]"].to_numpy()
    z = df["Z [m]"].to_numpy()
    vx = np.gradient(x, STEP_SECONDS)
    vy = np.gradient(y, STEP_SECONDS)
    vz = np.gradient(z, STEP_SECONDS)
    speed = df["Speed resulting [km/h]"].to_numpy() * (1000 / 3600)
    body = [speed, *[df[column].to_numpy() for column in BODY_COLUMNS]]
    states = np.stack([x, y, z, vx, vy, vz, *body], axis=1)
    observations = np.stack(
        [x, y, z, df["Height above ground [m]"].to_numpy()], axis=1
    )

    wind_controls = []
    for feature in WIND_FEATURES:
        for zone in ZONES:
            columns = [f"{sensor}_{feature}" for sensor in zone]
            wind_controls.append(df[columns].mean(axis=1).to_numpy())
    controls = np.stack([*wind_controls, *body], axis=1)
    return states, observations, controls, roll_missing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(RAW_DIR.glob("*.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw CSV files found under {RAW_DIR}")

    states: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    accepted_filenames: list[str] = []
    audit_rows: list[dict[str, object]] = []

    for path in raw_paths:
        flight = flatten_raw_file(path)
        original_time = flight["Time"].to_numpy(dtype=float)
        repaired_time, repaired_indices = repair_timestamps(original_time)
        normalized = resample(flight, repaired_time)
        normalized_name = f"{path.stem}_cleaned_normalized.csv"
        normalized.to_csv(OUTPUT_DIR / normalized_name, index=False)

        x_state, y_observation, u_control, roll_missing = build_model_arrays(
            normalized, path.name
        )
        accepted = path.name != EXCLUDED_FILE
        if accepted:
            states.append(x_state)
            observations.append(y_observation)
            controls.append(u_control)
            accepted_filenames.append(path.name)

        differences = np.diff(original_time)
        audit_rows.append(
            {
                "filename": path.name,
                "accepted_for_model": accepted,
                "source_rows_after_preflight": len(flight),
                "source_timestamp_decreases": int(np.sum(differences < 0)),
                "source_timestamp_duplicates": int(np.sum(differences == 0)),
                "timestamps_repaired": len(repaired_indices),
                "maximum_absolute_adjustment_seconds": (
                    float(np.max(np.abs(repaired_time - original_time)))
                    if len(repaired_indices)
                    else 0.0
                ),
                "resampled_rows": len(normalized),
                "roll_cells_filled": roll_missing,
            }
        )

    lengths = np.asarray([len(sequence) for sequence in states])
    np.savez(
        OUTPUT_DIR / "combined_dataset.npz",
        X=np.asarray(states, dtype=object),
        Y=np.asarray(observations, dtype=object),
        U=np.asarray(controls, dtype=object),
        filenames=np.asarray(accepted_filenames),
    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT_DIR / "timestamp_repair_audit.csv", index=False)
    pd.DataFrame({"filename": accepted_filenames}).to_csv(
        OUTPUT_DIR / "accepted_files.csv", index=False
    )

    summary = {
        "raw_files": len(raw_paths),
        "accepted_files": len(states),
        "excluded_file": EXCLUDED_FILE,
        "files_with_source_timestamp_decreases": int(
            (audit["source_timestamp_decreases"] > 0).sum()
        ),
        "source_timestamp_decreases": int(
            audit["source_timestamp_decreases"].sum()
        ),
        "files_with_source_timestamp_duplicates": int(
            (audit["source_timestamp_duplicates"] > 0).sum()
        ),
        "source_timestamp_duplicates": int(
            audit["source_timestamp_duplicates"].sum()
        ),
        "files_with_timestamp_repairs": int(
            (audit["timestamps_repaired"] > 0).sum()
        ),
        "timestamps_repaired": int(audit["timestamps_repaired"].sum()),
        "roll_cells_filled": int(audit["roll_cells_filled"].sum()),
        "sample_interval_seconds": STEP_SECONDS,
        "samples_min": int(lengths.min()),
        "samples_median": float(np.median(lengths)),
        "samples_mean": float(lengths.mean()),
        "samples_max": int(lengths.max()),
        "duration_seconds_min": float((lengths.min() - 1) * STEP_SECONDS),
        "duration_seconds_median": float((np.median(lengths) - 1) * STEP_SECONDS),
        "duration_seconds_mean": float((lengths.mean() - 1) * STEP_SECONDS),
        "duration_seconds_max": float((lengths.max() - 1) * STEP_SECONDS),
    }
    (OUTPUT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote corrected dataset and audit files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
