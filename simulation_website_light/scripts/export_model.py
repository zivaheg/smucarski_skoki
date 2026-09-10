"""Export the production ski-jump SSM as self-contained browser assets.

This script does not retrain the model. It converts the audited CSV matrices,
baseline state sequence, and hill profile used by ``skijump_app`` into versioned
JSON files consumed by the static website.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


WEBSITE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = WEBSITE_DIR.parent
SOURCE_DIR = PROJECT_DIR / "skijump_app"
OUTPUT_DIR = WEBSITE_DIR / "public" / "data"

STATE_NAMES = [
    ("x", "X position", "m"),
    ("y", "Y position", "m"),
    ("z", "Z position", "m"),
    ("vx", "X velocity", "m/s"),
    ("vy", "Y velocity", "m/s"),
    ("vz", "Z velocity", "m/s"),
    ("speed", "Resulting speed", "m/s"),
    ("opening", "Opening angle", "deg"),
    ("rollLeft", "Left roll angle", "deg"),
    ("rollRight", "Right roll angle", "deg"),
    ("yawLeft", "Left yaw angle", "deg"),
    ("yawRight", "Right yaw angle", "deg"),
    ("stallLeft", "Left stalling angle", "deg"),
    ("stallRight", "Right stalling angle", "deg"),
]

CONTROL_NAMES = [
    ("windSpeedTakeoff", "Wind speed", "m/s", "takeoff", "offset"),
    ("windSpeedMiddle", "Wind speed", "m/s", "middle", "offset"),
    ("windSpeedLanding", "Wind speed", "m/s", "landing", "offset"),
    ("windTangentTakeoff", "Tangent wind", "m/s", "takeoff", "offset"),
    ("windTangentMiddle", "Tangent wind", "m/s", "middle", "offset"),
    ("windTangentLanding", "Tangent wind", "m/s", "landing", "offset"),
    ("crosswindTakeoff", "Crosswind", "m/s", "takeoff", "offset"),
    ("crosswindMiddle", "Crosswind", "m/s", "middle", "offset"),
    ("crosswindLanding", "Crosswind", "m/s", "landing", "offset"),
    ("turbulenceTakeoff", "Turbulence", "m/s", "takeoff", "offset"),
    ("turbulenceMiddle", "Turbulence", "m/s", "middle", "offset"),
    ("turbulenceLanding", "Turbulence", "m/s", "landing", "offset"),
    ("speed", "Mean speed", "m/s", "body", "mean"),
    ("opening", "Mean opening angle", "deg", "body", "mean"),
    ("rollLeft", "Mean left roll", "deg", "body", "mean"),
    ("rollRight", "Mean right roll", "deg", "body", "mean"),
    ("yawLeft", "Mean left yaw", "deg", "body", "mean"),
    ("yawRight", "Mean right yaw", "deg", "body", "mean"),
    ("stallLeft", "Mean left stall", "deg", "body", "mean"),
    ("stallRight", "Mean right stall", "deg", "body", "mean"),
]

OBSERVATIONS = [
    ("x", "X position", "m"),
    ("y", "Y position", "m"),
    ("z", "Z position", "m"),
    ("height", "Height above ground", "m"),
]

BASELINE_WIND_MEANS = [
    1.0747,
    1.3615,
    1.0120,
    0.5363,
    0.5560,
    0.1634,
    0.5514,
    0.7760,
    0.4583,
    0.1245,
    0.1821,
    0.1360,
]

WIND_OFFSET_LIMITS = [
    [-0.6, 0.9],
    [-0.8, 1.0],
    [-0.6, 1.2],
    [-1.0, 1.4],
    [-1.6, 1.5],
    [-1.5, 1.7],
    [-0.4, 0.5],
    [-0.6, 1.1],
    [-0.3, 0.5],
    [-0.07, 0.13],
    [-0.10, 0.19],
    [-0.08, 0.12],
]

BODY_MEAN_LIMITS = [
    [28.8, 33.4],
    [14.9, 39.1],
    [13.7, 34.9],
    [-36.0, -15.8],
    [3.7, 21.1],
    [-21.6, -5.3],
    [-7.7, 5.3],
    [-6.9, 7.1],
]


def load_matrix(name: str) -> np.ndarray:
    return np.loadtxt(SOURCE_DIR / name, delimiter=",")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(name: str, value: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / name
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.relative_to(PROJECT_DIR)}")


def simulate(
    a: np.ndarray,
    b: np.ndarray,
    baseline: np.ndarray,
    sliders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_profiles = baseline[:, 6:14]
    body_means = body_profiles.mean(axis=0)
    controls = np.empty((len(baseline), 20), dtype=float)
    controls[:, :12] = np.asarray(BASELINE_WIND_MEANS) + sliders[:12]
    controls[:, 12:20] = body_profiles + (sliders[12:20] - body_means)
    controls[:, 13:20] += sliders[20]

    states = [baseline[0]]
    for index in range(len(baseline) - 1):
        states.append(a @ states[-1] + b @ controls[index])
    return np.asarray(states), controls


def main() -> None:
    source_files = {
        "A": SOURCE_DIR / "matrixA_AVG.csv",
        "B": SOURCE_DIR / "matrixB_AVG.csv",
        "C": SOURCE_DIR / "matrixC_AVG.csv",
        "D": SOURCE_DIR / "matrixD_AVG.csv",
        "baseline": SOURCE_DIR / "xImportant_AVG.csv",
        "hill": SOURCE_DIR / "hill_profile.csv",
    }
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source assets: {missing}")

    a = load_matrix("matrixA_AVG.csv")
    b = load_matrix("matrixB_AVG.csv")
    c = load_matrix("matrixC_AVG.csv")
    d = load_matrix("matrixD_AVG.csv")
    baseline = load_matrix("xImportant_AVG.csv")

    expected = [(14, 14), (14, 20), (4, 14), (4, 20), (139, 14)]
    actual = [a.shape, b.shape, c.shape, d.shape, baseline.shape]
    if actual != expected:
        raise ValueError(f"Unexpected model dimensions: {actual}; expected {expected}")

    body_profiles = baseline[:, 6:14]
    body_means = body_profiles.mean(axis=0)
    default_body = np.round(body_means, 1)
    default_sliders = np.concatenate([np.zeros(12), default_body, np.zeros(1)])
    states, controls = simulate(a, b, baseline, default_sliders)
    endpoint = states[-1, :3]
    displayed_distance = math.hypot(endpoint[0], endpoint[2])

    step_sizes = [0.1] * 9 + [0.01] * 3 + [0.1] * 8
    slider_limits = WIND_OFFSET_LIMITS + BODY_MEAN_LIMITS
    control_metadata = []
    for index, ((key, label, unit, group, value_kind), limits, step) in enumerate(
        zip(CONTROL_NAMES, slider_limits, step_sizes)
    ):
        control_metadata.append(
            {
                "index": index,
                "key": key,
                "label": label,
                "unit": unit,
                "group": group,
                "valueKind": value_kind,
                "min": limits[0],
                "max": limits[1],
                "step": step,
                "default": float(default_sliders[index]),
                "baseline": (
                    BASELINE_WIND_MEANS[index]
                    if index < 12
                    else float(body_means[index - 12])
                ),
            }
        )

    model = {
        "schemaVersion": 1,
        "modelName": "Planica 20-control average-flight SSM",
        "description": "Discrete linear state-space model fitted to accepted normalized ski-jump sequences.",
        "sampleIntervalSeconds": 0.05,
        "trainedHorizon": len(baseline),
        "training": {
            "acceptedSequences": 203,
            "excludedSequenceIndex": 169,
            "regression": "Ridge",
            "alpha": 10.0,
            "fitIntercept": False,
            "reportedCrossValidationErrorMetres": 1.625906478498713,
            "reportedAverageFlightBaselineErrorMetres": 2.7208366614042823,
        },
        "states": [
            {"index": index, "key": key, "label": label, "unit": unit}
            for index, (key, label, unit) in enumerate(STATE_NAMES)
        ],
        "controls": control_metadata,
        "observations": [
            {"index": index, "key": key, "label": label, "unit": unit}
            for index, (key, label, unit) in enumerate(OBSERVATIONS)
        ],
        "globalAngleOffset": {
            "index": 20,
            "key": "overallAngleOffset",
            "label": "Overall angle offset",
            "unit": "deg",
            "min": -1.5,
            "max": 1.5,
            "step": 0.1,
            "default": 0.0,
            "appliesToControlIndices": list(range(13, 20)),
        },
        "baselineWindMeans": BASELINE_WIND_MEANS,
        "matrices": {
            "A": a.tolist(),
            "B": b.tolist(),
            "C": c.tolist(),
            "D": d.tolist(),
        },
        "diagnostics": {
            "spectralRadiusA": float(np.max(np.abs(np.linalg.eigvals(a)))),
            "rankA": int(np.linalg.matrix_rank(a)),
            "rankB": int(np.linalg.matrix_rank(b)),
            "rankC": int(np.linalg.matrix_rank(c)),
        },
        "warnings": [
            "Outputs are predictions of a fitted linear model, not causal physical conclusions.",
            "Speed and body-angle values occur in both state and control vectors.",
            "Do not extrapolate beyond the trained 139-point flight horizon.",
            "Several individually valid slider extremes can form an unusual combined scenario.",
        ],
        "provenance": {
            key: {
                "source": str(path.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for key, path in source_files.items()
        },
    }

    baseline_asset = {
        "schemaVersion": 1,
        "sampleIntervalSeconds": 0.05,
        "initialState": baseline[0].tolist(),
        "states": baseline.tolist(),
        "bodyProfiles": body_profiles.tolist(),
        "bodyMeans": body_means.tolist(),
        "defaultBodySliderValues": default_body.tolist(),
        "defaultSliderValues": default_sliders.tolist(),
        "referenceSimulation": {
            "endpoint": endpoint.tolist(),
            "displayedDistance": displayed_distance,
            "maxZ": float(states[:, 2].max()),
            "maxAbsY": float(np.abs(states[:, 1]).max()),
            "finalState": states[-1].tolist(),
            "firstControl": controls[0].tolist(),
        },
    }

    with source_files["hill"].open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        hill_points = [
            [float(row["X [m]"]), float(row["Ground Z [m]"])] for row in rows
        ]

    hill_asset = {
        "schemaVersion": 1,
        "coordinateSystem": {
            "x": "forward along the jump [m]",
            "y": "lateral across the hill [m]",
            "z": "vertical elevation relative to takeoff [m]",
        },
        "surfaceHalfWidth": 18.0,
        "distanceMarkers": list(range(25, 176, 25)),
        "points": hill_points,
        "note": "Longitudinal measured profile extruded laterally for visualization; not a surveyed 3D terrain mesh.",
    }

    write_json("model.json", model)
    write_json("baseline-flight.json", baseline_asset)
    write_json("hill-profile.json", hill_asset)
    print(
        "Reference endpoint "
        f"({endpoint[0]:.9f}, {endpoint[1]:.9f}, {endpoint[2]:.9f}), "
        f"displayed distance {displayed_distance:.9f} m"
    )


if __name__ == "__main__":
    main()
