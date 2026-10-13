"""Compare wind feature designs on the corrected timestamp dataset.

All variants use the manuscript's 14-state formulation, measured initial state,
Ridge(alpha=10, fit_intercept=False), and jump-level leave-one-out folds.  Only
the representation of wind in the control vector changes.  The body controls
are copied from the corrected production archive so that preprocessing is
identical across variants.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from revision_analysis import (
    ALPHA,
    DATA,
    interval_95,
    landing_distance,
    simulate,
    summarize,
    trajectory_error,
)


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_DIR = DATA.parent
OUTPUT_JSON = ROOT / "simulation" / "wind_feature_sensitivity.json"
OUTPUT_CSV = ROOT / "simulation" / "wind_feature_sensitivity_per_jump.csv"
WIND_FEATURES = ("Speed", "Tangent", "Cross", "Turbulence")
ZONES = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))


def load_wind_controls(
    filenames: list[str], production_controls: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return aligned zone-mean and individual-sensor control sequences."""
    zone_controls: list[np.ndarray] = []
    sensor_controls: list[np.ndarray] = []
    for filename, production in zip(filenames, production_controls):
        normalized_path = CORRECTED_DIR / (
            f"{Path(filename).stem}_cleaned_normalized.csv"
        )
        frame = pd.read_csv(normalized_path)
        individual = np.column_stack(
            [
                frame[f"W{sensor}_{feature}"].to_numpy(dtype=float)
                for feature in WIND_FEATURES
                for sensor in range(1, 13)
            ]
        )
        zone = np.column_stack(
            [
                frame[
                    [f"W{sensor}_{feature}" for sensor in sensors]
                ].mean(axis=1).to_numpy(dtype=float)
                for feature in WIND_FEATURES
                for sensors in ZONES
            ]
        )
        if len(zone) != len(production):
            raise ValueError(f"Length mismatch for {filename}")
        if not np.allclose(zone, production[:, :12], atol=1e-12, rtol=0.0):
            raise ValueError(f"Zone controls do not match production data: {filename}")
        zone_controls.append(zone)
        sensor_controls.append(individual)
    return zone_controls, sensor_controls


def fit_model(
    states: list[np.ndarray], controls: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    x_now = np.vstack([state[:-1] for state in states])
    x_next = np.vstack([state[1:] for state in states])
    u_now = np.vstack([control[:-1] for control in controls])
    regression = Ridge(alpha=ALPHA, fit_intercept=False)
    regression.fit(np.hstack((x_now, u_now)), x_next)
    coefficients = regression.coef_
    return coefficients[:, :14], coefficients[:, 14:]


def paired_summary(values: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "ci95_t": list(interval_95(values)),
        "fraction_variant_lower": float(np.mean(values < 0)),
    }


def main() -> None:
    archive = np.load(DATA, allow_pickle=True)
    states = [np.asarray(value, dtype=float) for value in archive["X"]]
    production = [np.asarray(value, dtype=float) for value in archive["U"]]
    filenames = [str(value) for value in archive["filenames"]]
    if len(states) != 203:
        raise ValueError("Corrected archive must contain 203 accepted jumps")

    zone, sensors = load_wind_controls(filenames, production)
    body = [control[:, 12:] for control in production]
    variants = {
        "zone_linear": [
            np.column_stack((wind, posture)) for wind, posture in zip(zone, body)
        ],
        "individual_linear": [
            np.column_stack((wind, posture)) for wind, posture in zip(sensors, body)
        ],
        "zone_linear_and_squared": [
            np.column_stack((wind, np.square(wind), posture))
            for wind, posture in zip(zone, body)
        ],
        "individual_linear_and_squared": [
            np.column_stack((wind, np.square(wind), posture))
            for wind, posture in zip(sensors, body)
        ],
    }

    records: list[dict[str, object]] = []
    by_variant: dict[str, dict[str, np.ndarray]] = {}
    for variant_name, controls in variants.items():
        print(f"Evaluating {variant_name} ({controls[0].shape[1]} controls)")
        metrics = {
            "trajectory_error_m": [],
            "endpoint_error_m": [],
            "terminal_coordinate_distance_absolute_error_m": [],
            "spectral_radius": [],
        }
        for held_out in range(len(states)):
            train_states = states[:held_out] + states[held_out + 1 :]
            train_controls = controls[:held_out] + controls[held_out + 1 :]
            a, b = fit_model(train_states, train_controls)
            actual = states[held_out]
            predicted = simulate(a, b, actual[0], controls[held_out])
            values = {
                "trajectory_error_m": trajectory_error(
                    actual[:, :3], predicted[:, :3]
                ),
                "endpoint_error_m": float(
                    np.linalg.norm(actual[-1, :3] - predicted[-1, :3])
                ),
                "terminal_coordinate_distance_absolute_error_m": abs(
                    landing_distance(actual[:, :3])
                    - landing_distance(predicted[:, :3])
                ),
                "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
            }
            for metric, value in values.items():
                metrics[metric].append(value)
            records.append(
                {
                    "variant": variant_name,
                    "control_dimension": controls[0].shape[1],
                    "jump_index": held_out,
                    "filename": filenames[held_out],
                    **values,
                }
            )
        by_variant[variant_name] = {
            metric: np.asarray(values, dtype=float) for metric, values in metrics.items()
        }

    reference = by_variant["zone_linear"]
    summary: dict[str, object] = {
        "description": (
            "Post-hoc wind-feature sensitivity on timestamp-corrected data using "
            "fixed alpha=10 and identical jump-level LOOCV folds"
        ),
        "configuration": {
            "accepted_jumps": len(states),
            "state_dimension": 14,
            "ridge_alpha": ALPHA,
            "fit_intercept": False,
            "initial_state": "measured held-out initial state",
            "wind_order": "feature-major, then zone or sensor",
        },
        "variants": {},
    }
    variants_summary = summary["variants"]
    assert isinstance(variants_summary, dict)
    for name, metrics in by_variant.items():
        comparisons = {}
        if name != "zone_linear":
            comparisons = {
                metric: paired_summary(values - reference[metric])
                for metric, values in metrics.items()
            }
        variants_summary[name] = {
            "control_dimension": variants[name][0].shape[1],
            "metrics": {
                metric: summarize(values.tolist()) for metric, values in metrics.items()
            },
            "paired_difference_vs_zone_linear": comparisons,
        }

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
