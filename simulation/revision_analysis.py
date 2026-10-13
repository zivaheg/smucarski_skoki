"""Reproducible evaluation used for the revised Informatica manuscript.

The script deliberately keeps every preprocessing decision needed for evaluation
in one place.  In particular, all data-derived initial states and average-path
baselines are recomputed from the training jumps of each LOOCV fold; the held-out
jump is never used to construct them.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "simulation"
    / "2024_03_Planica_12_winds"
    / "cleaned"
    / "normalized_corrected"
    / "combined_dataset.npz"
)
OUTPUT_JSON = ROOT / "simulation" / "revision_metrics.json"
OUTPUT_CSV = ROOT / "simulation" / "revision_per_jump_metrics.csv"
OUTPUT_FIGURE = ROOT / "revision_trajectory_representative.pdf"
OUTPUT_FIGURE_PNG = ROOT / "revision_trajectory_representative.png"
OUTPUT_FIGURE_METADATA = ROOT / "simulation" / "revision_trajectory_representative.json"

ALPHA = 10.0
SAMPLE_INTERVAL_SECONDS = 0.05
RESAMPLED_PATH_POINTS = 1000


def merge_duplicate_x(points: np.ndarray) -> np.ndarray:
    """Sort a 3-D path by x and average y/z for duplicate x values."""
    order = np.argsort(points[:, 0], kind="stable")
    points = np.asarray(points, dtype=float)[order]
    unique_x, inverse = np.unique(points[:, 0], return_inverse=True)
    merged = np.zeros((len(unique_x), 3), dtype=float)
    counts = np.bincount(inverse)
    merged[:, 0] = unique_x
    merged[:, 1] = np.bincount(inverse, weights=points[:, 1]) / counts
    merged[:, 2] = np.bincount(inverse, weights=points[:, 2]) / counts
    return merged


def interpolate_path_at_x(points: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    points = merge_duplicate_x(points)
    return np.column_stack(
        (
            x_values,
            np.interp(x_values, points[:, 0], points[:, 1]),
            np.interp(x_values, points[:, 0], points[:, 2]),
        )
    )


def trajectory_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Euclidean separation after alignment at integer longitudinal x.

    Both paths are sorted by x.  Their first points, all integer x locations in
    the overlapping range, and the endpoint of the shorter path are compared.
    Beyond the overlap, points on the longer path are compared with the shorter
    path's final point, thereby penalizing an incorrect predicted path length.
    """
    actual_raw = np.asarray(actual, dtype=float)[:, :3]
    predicted_raw = np.asarray(predicted, dtype=float)[:, :3]
    actual = merge_duplicate_x(actual_raw)
    predicted = merge_duplicate_x(predicted_raw)
    distances = [float(np.linalg.norm(actual_raw[0] - predicted_raw[0]))]

    actual_integer_x = np.arange(math.ceil(actual[0, 0]), math.floor(actual[-1, 0]) + 1)
    predicted_integer_x = np.arange(
        math.ceil(predicted[0, 0]), math.floor(predicted[-1, 0]) + 1
    )
    common_x = np.intersect1d(actual_integer_x, predicted_integer_x)
    common_x = common_x[common_x > max(actual_raw[0, 0], predicted_raw[0, 0])]
    if len(common_x):
        actual_common = interpolate_path_at_x(actual, common_x)
        predicted_common = interpolate_path_at_x(predicted, common_x)
        distances.extend(np.linalg.norm(actual_common - predicted_common, axis=1))

    if actual[-1, 0] < predicted[-1, 0]:
        tail_x = np.arange(math.floor(actual[-1, 0]) + 1, math.floor(predicted[-1, 0]) + 1)
        if len(tail_x):
            predicted_tail = interpolate_path_at_x(predicted, tail_x)
            distances.extend(np.linalg.norm(predicted_tail - actual[-1], axis=1))
        distances.append(float(np.linalg.norm(predicted[-1] - actual[-1])))
    elif actual[-1, 0] > predicted[-1, 0]:
        tail_x = np.arange(math.floor(predicted[-1, 0]) + 1, math.floor(actual[-1, 0]) + 1)
        if len(tail_x):
            actual_tail = interpolate_path_at_x(actual, tail_x)
            distances.extend(np.linalg.norm(actual_tail - predicted[-1], axis=1))
        distances.append(float(np.linalg.norm(actual[-1] - predicted[-1])))

    return float(np.mean(distances))


def fit_state_model(states: list[np.ndarray], controls: list[np.ndarray]):
    x_now = np.vstack([x[:-1] for x in states])
    x_next = np.vstack([x[1:] for x in states])
    u_now = np.vstack([u[:-1] for u in controls])
    regression = Ridge(alpha=ALPHA, fit_intercept=False)
    regression.fit(np.hstack((x_now, u_now)), x_next)
    coefficients = regression.coef_
    return coefficients[:, :14], coefficients[:, 14:]


def simulate(a: np.ndarray, b: np.ndarray, initial: np.ndarray, controls: np.ndarray):
    predicted = np.empty((len(controls), len(initial)), dtype=float)
    predicted[0] = initial
    for t in range(len(controls) - 1):
        predicted[t + 1] = a @ predicted[t] + b @ controls[t]
    return predicted


def normalized_resample(path: np.ndarray, n_points: int) -> np.ndarray:
    old_progress = np.linspace(0.0, 1.0, len(path))
    new_progress = np.linspace(0.0, 1.0, n_points)
    return np.column_stack(
        [np.interp(new_progress, old_progress, path[:, axis]) for axis in range(3)]
    )


def interval_95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    half_width = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(np.mean(values) - half_width), float(np.mean(values) + half_width)


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_std": float(np.std(array, ddof=1)),
        "ci95_t": list(interval_95(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def landing_distance(path: np.ndarray) -> float:
    return float(np.hypot(path[-1, 0], path[-1, 2]))


def main() -> None:
    archive = np.load(DATA, allow_pickle=True)
    states = list(archive["X"])
    controls = list(archive["U"])
    filenames = [str(value) for value in archive["filenames"]]
    n_jumps = len(states)
    if n_jumps != 203 or len(controls) != n_jumps or len(filenames) != n_jumps:
        raise ValueError("Corrected archive must contain 203 aligned accepted jumps")

    resampled_paths = np.stack(
        [normalized_resample(x[:, :3], RESAMPLED_PATH_POINTS) for x in states]
    )
    resampled_sum = np.sum(resampled_paths, axis=0)

    records: list[dict[str, float | int]] = []
    actual_start_predictions: list[np.ndarray] = []
    for held_out in range(n_jumps):
        train_ids = [i for i in range(n_jumps) if i != held_out]
        train_states = [states[i] for i in train_ids]
        train_controls = [controls[i] for i in train_ids]
        a, b = fit_state_model(train_states, train_controls)

        actual = states[held_out]
        u_test = controls[held_out]
        predicted_actual = simulate(a, b, actual[0], u_test)
        mean_initial = np.mean([x[0] for x in train_states], axis=0)
        predicted_mean = simulate(a, b, mean_initial, u_test)

        initial_velocity = actual[0, 3:6]
        elapsed = np.arange(len(actual))[:, None] * SAMPLE_INTERVAL_SECONDS
        constant_velocity = actual[0, :3] + elapsed * initial_velocity

        fold_average_1000 = (resampled_sum - resampled_paths[held_out]) / (n_jumps - 1)
        fold_mean_length = int(round(np.mean([len(states[i]) for i in train_ids])))
        average_path = normalized_resample(fold_average_1000, fold_mean_length)

        record = {
            "jump_index": held_out,
            "filename": filenames[held_out],
            "samples": len(actual),
            "duration_seconds": (len(actual) - 1) * SAMPLE_INTERVAL_SECONDS,
            "trajectory_error_actual_start_m": trajectory_error(
                actual[:, :3], predicted_actual[:, :3]
            ),
            "trajectory_error_training_mean_start_m": trajectory_error(
                actual[:, :3], predicted_mean[:, :3]
            ),
            "trajectory_error_constant_velocity_m": trajectory_error(
                actual[:, :3], constant_velocity
            ),
            "trajectory_error_training_average_path_m": trajectory_error(
                actual[:, :3], average_path
            ),
            "endpoint_error_actual_start_m": float(
                np.linalg.norm(actual[-1, :3] - predicted_actual[-1, :3])
            ),
            "landing_distance_absolute_error_actual_start_m": abs(
                landing_distance(actual[:, :3])
                - landing_distance(predicted_actual[:, :3])
            ),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
        }
        records.append(record)
        actual_start_predictions.append(predicted_actual)

    metric_names = [
        key
        for key in records[0]
        if key not in {"jump_index", "filename", "samples", "duration_seconds"}
    ]
    summary = {name: summarize([float(row[name]) for row in records]) for name in metric_names}
    lengths = np.asarray([len(x) for x in states])
    durations = (lengths - 1) * SAMPLE_INTERVAL_SECONDS
    summary["dataset"] = {
        "accepted_jumps": n_jumps,
        "samples_min": int(np.min(lengths)),
        "samples_median": float(np.median(lengths)),
        "samples_mean": float(np.mean(lengths)),
        "samples_max": int(np.max(lengths)),
        "duration_seconds_min": float(np.min(durations)),
        "duration_seconds_median": float(np.median(durations)),
        "duration_seconds_mean": float(np.mean(durations)),
        "duration_seconds_max": float(np.max(durations)),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "source_archive": str(DATA.relative_to(ROOT)).replace("\\", "/"),
    }
    summary["configuration"] = {
        "ridge_alpha": ALPHA,
        "fit_intercept": False,
        "state_dimension": 14,
        "control_dimension": 20,
        "resampled_average_path_points": RESAMPLED_PATH_POINTS,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    errors = np.asarray([row["trajectory_error_actual_start_m"] for row in records])
    actual_distances = np.asarray([landing_distance(x[:, :3]) for x in states])
    distance_iqr = float(np.subtract(*np.quantile(actual_distances, [0.75, 0.25])))
    error_iqr = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    median_error = float(np.median(errors))
    representative: dict[str, int] = {}
    for label, quantile in (("short", 0.10), ("typical", 0.50), ("long", 0.90)):
        target_distance = float(np.quantile(actual_distances, quantile))
        # Balance proximity to the requested distance quantile with proximity to
        # median model error.  This avoids cherry-picking unusually good or bad
        # simulations merely to obtain visually different jump lengths.
        score = (
            np.abs(actual_distances - target_distance) / distance_iqr
            + np.abs(errors - median_error) / error_iqr
        )
        representative[label] = int(np.argmin(score))

    figure_metadata = {
        "selection_rule": (
            "Closest joint match to the 10th, 50th, or 90th percentile of measured "
            "terminal distance and the median held-out trajectory error, scaled by "
            "their respective interquartile ranges"
        ),
        "dataset_terminal_distance_quantiles_m": {
            "q10": float(np.quantile(actual_distances, 0.10)),
            "q50": float(np.quantile(actual_distances, 0.50)),
            "q90": float(np.quantile(actual_distances, 0.90)),
        },
        "examples": {
            label: {
                "jump_index": index,
                "filename": filenames[index],
                "measured_terminal_distance_m": float(actual_distances[index]),
                "trajectory_error_m": float(errors[index]),
            }
            for label, index in representative.items()
        },
    }
    with OUTPUT_FIGURE_METADATA.open("w", encoding="utf-8") as handle:
        json.dump(figure_metadata, handle, indent=2)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 5.8),
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    for column, (label, index) in enumerate(representative.items()):
        actual = states[index]
        predicted = actual_start_predictions[index]
        path_axis = axes[0, column]
        error_axis = axes[1, column]
        path_axis.plot(
            actual[:, 0], actual[:, 2], color="black", lw=1.5, label="measured"
        )
        path_axis.plot(
            predicted[:, 0],
            predicted[:, 2],
            color="#0072B2",
            lw=1.5,
            ls="--",
            label="SSM",
        )
        path_axis.set_title(
            f"{label.capitalize()}: $d_{{\\mathrm{{coord}}}}$={actual_distances[index]:.1f} m, "
            f"$E$={errors[index]:.2f} m"
        )
        path_axis.set_xlabel("X (m)")
        path_axis.grid(alpha=0.25)

        actual_progress = normalized_resample(actual[:, :3], 101)
        predicted_progress = normalized_resample(predicted[:, :3], 101)
        pointwise_error = np.linalg.norm(actual_progress - predicted_progress, axis=1)
        error_axis.plot(
            np.linspace(0.0, 100.0, len(pointwise_error)),
            pointwise_error,
            color="#D55E00",
            lw=1.5,
        )
        error_axis.set_xlabel("Flight progress (%)")
        error_axis.set_ylim(bottom=0.0)
        error_axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Z (m)")
    axes[1, 0].set_ylabel("Progress-aligned error (m)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(f"Wrote {OUTPUT_FIGURE_PNG}")
    print(f"Wrote {OUTPUT_FIGURE_METADATA}")


if __name__ == "__main__":
    main()
