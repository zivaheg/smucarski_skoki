"""Export the production SSM used by the slider app.

This follows the training setup validated by fit_ssm_cv_AVG in notebook 10:
the non-quadratic 20-control dataset, Ridge(alpha=10), and no intercept. Cross
validation estimates performance; the production matrices are then fitted on
all accepted sequences so the app can use one model.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


ALPHA = 10.0
STATE_DIM = 14
EXCLUDED_SEQUENCE_INDEX = 169  # Same problematic sequence removed in notebook 10.

SIMULATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SIMULATION_DIR.parent
DATASET_PATH = (
    SIMULATION_DIR
    / "2024_03_Planica_12_winds"
    / "cleaned"
    / "normalized"
    / "combined_dataset.npz"
)
APP_DIR = PROJECT_DIR / "skijump_app"


def load_training_sequences():
    dataset = np.load(DATASET_PATH, allow_pickle=True)
    states = list(dataset["X"])
    observations = list(dataset["Y"])
    controls = list(dataset["U"])

    states.pop(EXCLUDED_SEQUENCE_INDEX)
    observations.pop(EXCLUDED_SEQUENCE_INDEX)
    controls.pop(EXCLUDED_SEQUENCE_INDEX)
    return states, observations, controls


def fit_production_model(states, observations, controls):
    current_states = np.vstack([sequence[:-1] for sequence in states])
    next_states = np.vstack([sequence[1:] for sequence in states])
    current_controls = np.vstack([sequence[:-1] for sequence in controls])
    current_observations = np.vstack(
        [sequence[:-1] for sequence in observations]
    )
    predictors = np.hstack([current_states, current_controls])

    state_model = Ridge(alpha=ALPHA, fit_intercept=False)
    state_model.fit(predictors, next_states)
    state_coefficients = state_model.coef_.T
    A = state_coefficients[:STATE_DIM].T
    B = state_coefficients[STATE_DIM:].T

    observation_model = Ridge(alpha=ALPHA, fit_intercept=False)
    observation_model.fit(predictors, current_observations)
    observation_coefficients = observation_model.coef_.T
    C = observation_coefficients[:STATE_DIM].T
    D = observation_coefficients[STATE_DIM:].T
    return A, B, C, D, len(current_states)


def export_matrices(matrices):
    for name, matrix in zip("ABCD", matrices):
        output_path = APP_DIR / f"matrix{name}_AVG.csv"
        np.savetxt(output_path, matrix, delimiter=",")


def main():
    states, observations, controls = load_training_sequences()
    A, B, C, D, transition_count = fit_production_model(
        states, observations, controls
    )

    if A.shape != (14, 14) or B.shape != (14, 20):
        raise ValueError(f"Unexpected matrix shapes: A={A.shape}, B={B.shape}")

    export_matrices((A, B, C, D))
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    print(f"Exported A{A.shape}, B{B.shape}, C{C.shape}, D{D.shape}")
    print(f"Sequences: {len(states)}, transitions: {transition_count}")
    print(f"Ridge alpha: {ALPHA}, A spectral radius: {spectral_radius:.6f}")


if __name__ == "__main__":
    main()
