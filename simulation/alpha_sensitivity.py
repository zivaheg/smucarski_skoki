"""Post-hoc LOOCV sensitivity analysis for the ridge penalty.

This script intentionally does not select a new production value of alpha.
It evaluates a fixed logarithmic grid with the same corrected data and held-out
trajectory metric as ``revision_analysis.py`` so that the manuscript can report
whether the legacy value alpha=10 lies in a robust region.  Because the grid is
examined on the evaluation data, it is descriptive rather than nested model
selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from revision_analysis import DATA, landing_distance, simulate, trajectory_error


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "simulation" / "alpha_sensitivity.csv"
OUTPUT_JSON = ROOT / "simulation" / "alpha_sensitivity.json"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_model(
    states: list[np.ndarray], controls: list[np.ndarray], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    x_now = np.vstack([x[:-1] for x in states])
    x_next = np.vstack([x[1:] for x in states])
    u_now = np.vstack([u[:-1] for u in controls])
    regression = Ridge(alpha=alpha, fit_intercept=False)
    regression.fit(np.hstack((x_now, u_now)), x_next)
    coefficients = regression.coef_
    return coefficients[:, :14], coefficients[:, 14:]


def main() -> None:
    archive = np.load(DATA, allow_pickle=True)
    states = list(archive["X"])
    controls = list(archive["U"])
    if len(states) != 203 or len(controls) != 203:
        raise ValueError("Corrected archive must contain 203 aligned accepted jumps")

    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        trajectory_errors: list[float] = []
        endpoint_errors: list[float] = []
        distance_errors: list[float] = []
        spectral_radii: list[float] = []
        for held_out in range(len(states)):
            train_states = states[:held_out] + states[held_out + 1 :]
            train_controls = controls[:held_out] + controls[held_out + 1 :]
            a, b = fit_model(train_states, train_controls, alpha)
            actual = states[held_out]
            predicted = simulate(a, b, actual[0], controls[held_out])
            trajectory_errors.append(trajectory_error(actual[:, :3], predicted[:, :3]))
            endpoint_errors.append(
                float(np.linalg.norm(actual[-1, :3] - predicted[-1, :3]))
            )
            distance_errors.append(
                abs(landing_distance(actual[:, :3]) - landing_distance(predicted[:, :3]))
            )
            spectral_radii.append(float(np.max(np.abs(np.linalg.eigvals(a)))))

        row = {
            "alpha": alpha,
            "trajectory_mean_m": float(np.mean(trajectory_errors)),
            "trajectory_median_m": float(np.median(trajectory_errors)),
            "endpoint_mean_m": float(np.mean(endpoint_errors)),
            "terminal_distance_mae_m": float(np.mean(distance_errors)),
            "spectral_radius_mean": float(np.mean(spectral_radii)),
        }
        rows.append(row)
        print(json.dumps(row))

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "description": (
                    "Post-hoc, non-nested LOOCV sensitivity of the unstandardized "
                    "14-state/20-control model to the ridge penalty"
                ),
                "alphas": rows,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
