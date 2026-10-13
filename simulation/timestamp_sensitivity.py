"""Evaluate the final model only on jumps that required no timestamp repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_analysis import fit_state_model, simulate, summarize, trajectory_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "simulation"
    / "2024_03_Planica_12_winds"
    / "cleaned"
    / "normalized_corrected"
)
OUTPUT = ROOT / "simulation" / "timestamp_sensitivity_metrics.json"


def main() -> None:
    archive = np.load(DATA_DIR / "combined_dataset.npz", allow_pickle=True)
    states = list(archive["X"])
    controls = list(archive["U"])
    filenames = [str(value) for value in archive["filenames"]]
    audit = pd.read_csv(DATA_DIR / "timestamp_repair_audit.csv").set_index("filename")
    untouched = [
        index
        for index, filename in enumerate(filenames)
        if int(audit.loc[filename, "timestamps_repaired"]) == 0
    ]
    states = [states[index] for index in untouched]
    controls = [controls[index] for index in untouched]

    errors: list[float] = []
    for held_out in range(len(states)):
        training_states = states[:held_out] + states[held_out + 1 :]
        training_controls = controls[:held_out] + controls[held_out + 1 :]
        a, b = fit_state_model(training_states, training_controls)
        predicted = simulate(a, b, states[held_out][0], controls[held_out])
        errors.append(
            trajectory_error(states[held_out][:, :3], predicted[:, :3])
        )

    result = {
        "description": "LOOCV sensitivity restricted to accepted jumps requiring no timestamp repair",
        "jumps": len(states),
        "trajectory_error_actual_start_m": summarize(errors),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
