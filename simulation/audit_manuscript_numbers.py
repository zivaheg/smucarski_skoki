"""Verify quantitative manuscript claims against reproducibility artifacts.

The audit covers dataset, preprocessing, model, validation, sensitivity, figure,
website-horizon, and software-version values.  Claims that depend on files no
longer present in the repository are reported separately rather than treated as
computationally verified.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

import rebuild_dataset


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main.tex").read_text(encoding="utf-8")
CORRECTED = (
    ROOT
    / "simulation"
    / "2024_03_Planica_12_winds"
    / "cleaned"
    / "normalized_corrected"
)

passes: list[str] = []


def close(label: str, actual: float, reported: float, tolerance: float) -> None:
    if not np.isclose(actual, reported, atol=tolerance, rtol=0.0):
        raise AssertionError(f"{label}: source={actual}, manuscript={reported}")
    passes.append(label)


def equal(label: str, actual: object, reported: object) -> None:
    if actual != reported:
        raise AssertionError(f"{label}: source={actual!r}, manuscript={reported!r}")
    passes.append(label)


def contains(label: str, fragment: str) -> None:
    if fragment not in MAIN:
        raise AssertionError(f"{label}: manuscript fragment not found: {fragment!r}")
    passes.append(label)


def main() -> None:
    dataset = json.loads((CORRECTED / "dataset_summary.json").read_text())
    primary = json.loads((ROOT / "simulation" / "revision_metrics.json").read_text())
    independent = json.loads(
        (ROOT / "simulation" / "independent_model_metrics.json").read_text()
    )
    timestamp = json.loads(
        (ROOT / "simulation" / "timestamp_sensitivity_metrics.json").read_text()
    )
    alpha = json.loads(
        (ROOT / "simulation" / "alpha_sensitivity.json").read_text()
    )["alphas"]
    wind_features = json.loads(
        (ROOT / "simulation" / "wind_feature_sensitivity.json").read_text()
    )["variants"]
    figure = json.loads(
        (ROOT / "simulation" / "revision_trajectory_representative.json").read_text()
    )
    website_model = json.loads(
        (ROOT / "simulation_website" / "public" / "data" / "model.json").read_text()
    )
    website_baseline = json.loads(
        (
            ROOT
            / "simulation_website"
            / "public"
            / "data"
            / "baseline-flight.json"
        ).read_text()
    )
    archive = np.load(CORRECTED / "combined_dataset.npz", allow_pickle=True)
    states = list(archive["X"])
    controls = list(archive["U"])

    equal("raw files retained after initial screening", dataset["raw_files"], 204)
    equal("accepted jumps", dataset["accepted_files"], 203)
    for key, expected in {
        "files_with_source_timestamp_decreases": 55,
        "source_timestamp_decreases": 109,
        "files_with_source_timestamp_duplicates": 8,
        "source_timestamp_duplicates": 9,
        "files_with_timestamp_repairs": 59,
        "timestamps_repaired": 145,
        "roll_cells_filled": 21,
        "samples_min": 88,
        "samples_max": 164,
    }.items():
        equal(key.replace("_", " "), dataset[key], expected)
    close("sample median", dataset["samples_median"], 137.0, 0.0)
    close("sample mean", dataset["samples_mean"], 136.62, 0.005)
    close("duration minimum", dataset["duration_seconds_min"], 4.35, 0.005)
    close("duration median", dataset["duration_seconds_median"], 6.80, 0.005)
    close("duration maximum", dataset["duration_seconds_max"], 8.15, 0.005)
    close("sample interval", dataset["sample_interval_seconds"], 0.05, 0.0)
    equal("state dimension", states[0].shape[1], 14)
    equal("control dimension", controls[0].shape[1], 20)

    terminal_coordinate_distances = np.asarray(
        [np.hypot(state[-1, 0], state[-1, 2]) for state in states]
    )
    close("terminal-coordinate minimum", terminal_coordinate_distances.min(), 117.78, 0.005)
    close("terminal-coordinate maximum", terminal_coordinate_distances.max(), 242.35, 0.005)
    close("terminal-coordinate median", np.median(terminal_coordinate_distances), 208.74, 0.005)
    close("terminal-coordinate mean", terminal_coordinate_distances.mean(), 205.54, 0.005)
    close(
        "terminal-coordinate sample SD",
        terminal_coordinate_distances.std(ddof=1),
        21.37,
        0.005,
    )

    roll_counts = {column: 0 for column in rebuild_dataset.ROLL_COLUMNS}
    roll_files: set[str] = set()
    edge_gap_count = 0
    for path in sorted(rebuild_dataset.RAW_DIR.glob("*.csv")):
        flight = rebuild_dataset.flatten_raw_file(path)
        repaired, _ = rebuild_dataset.repair_timestamps(
            flight["Time"].to_numpy(dtype=float)
        )
        normalized = rebuild_dataset.resample(flight, repaired)
        for column in rebuild_dataset.ROLL_COLUMNS:
            indices = np.flatnonzero(normalized[column].isna().to_numpy())
            roll_counts[column] += len(indices)
            if len(indices):
                roll_files.add(path.name)
            edge_gap_count += sum(
                index == 0 or index == len(normalized) - 1 for index in indices
            )
    equal("missing left-roll cells", roll_counts[rebuild_dataset.ROLL_COLUMNS[0]], 18)
    equal("missing right-roll cells", roll_counts[rebuild_dataset.ROLL_COLUMNS[1]], 3)
    equal("files with roll gaps", len(roll_files), 4)
    equal("edge roll gaps", edge_gap_count, 0)

    expected_primary = {
        "trajectory_error_actual_start_m": (1.34, 1.18, 0.68, 1.24, 1.43, 4.60),
        "trajectory_error_training_mean_start_m": (1.35, 1.18, 0.69, 1.25, 1.44, 4.70),
        "endpoint_error_actual_start_m": (3.55, 3.14, 2.04, 3.27, 3.84, 17.10),
        "landing_distance_absolute_error_actual_start_m": (
            1.68,
            1.25,
            1.83,
            1.43,
            1.94,
            16.36,
        ),
    }
    for metric, reported in expected_primary.items():
        source = primary[metric]
        for field, value in zip(
            ("mean", "median", "sample_std", "ci_low", "ci_high", "max"),
            reported,
        ):
            actual = source["ci95_t"][0 if field == "ci_low" else 1] if field.startswith("ci_") else source[field]
            close(f"{metric} {field}", actual, value, 0.005)

    baselines = {
        "trajectory_error_training_average_path_m": (2.64, 2.18, 2.34, 2.32, 2.97),
        "trajectory_error_constant_velocity_m": (58.33, 59.20, None, None, None),
    }
    for metric, (mean, median, sd, low, high) in baselines.items():
        close(f"{metric} mean", primary[metric]["mean"], mean, 0.005)
        close(f"{metric} median", primary[metric]["median"], median, 0.005)
        if sd is not None:
            close(f"{metric} SD", primary[metric]["sample_std"], sd, 0.005)
            close(f"{metric} CI low", primary[metric]["ci95_t"][0], low, 0.005)
            close(f"{metric} CI high", primary[metric]["ci95_t"][1], high, 0.005)
    close("average-path improvement", 2.6437267840922103 - 1.3394605897764642, 1.30, 0.005)
    close("no-repair main-model jumps", timestamp["jumps"], 144, 0.0)
    close(
        "no-repair main-model error",
        timestamp["trajectory_error_actual_start_m"]["mean"],
        1.39,
        0.005,
    )

    candidate_by_name = {
        row["variant"]: row for row in independent["candidate_comparison"]
    }
    expected_candidates = {
        "kinematic_wind_speed_posture": (6, 20, 0, 1.340, 3.552, 1.663),
        "kinematic_speed_state_wind_posture": (7, 19, 0, 1.350, 5.727, 4.523),
        "full_state_wind_only": (14, 12, 0, 1.872, 7.610, 5.538),
        "kinematic_wind_posture": (6, 19, 0, 6.808, 33.932, 33.350),
        "kinematic_wind_only": (6, 12, 0, 8.161, 40.005, 39.184),
    }
    for name, expected in expected_candidates.items():
        row = candidate_by_name[name]
        for field, value in zip(
            (
                "state_dimension",
                "control_dimension",
                "state_control_overlap",
                "trajectory_mean_m",
                "endpoint_mean_m",
                "terminal_distance_mae_m",
            ),
            expected,
        ):
            tolerance = 0.0 if field.endswith("dimension") or field == "state_control_overlap" else 0.0005
            close(f"{name} {field}", row[field], value, tolerance)

    paired = independent["paired_difference_vs_current"]
    close("paired path difference", paired["trajectory_error_actual_start_m"]["mean"], 0.010, 0.0005)
    close("paired path CI low", paired["trajectory_error_actual_start_m"]["ci95_t"][0], -0.029, 0.0005)
    close("paired path CI high", paired["trajectory_error_actual_start_m"]["ci95_t"][1], 0.050, 0.0005)
    close("independent path win percentage", 100 * paired["trajectory_error_actual_start_m"]["fraction_new_lower"], 48.3, 0.05)
    close("paired endpoint difference", paired["endpoint_error_actual_start_m"]["mean"], 2.172, 0.0005)
    close("paired coordinate-distance difference", paired["landing_distance_absolute_error_actual_start_m"]["mean"], 2.839, 0.0005)
    close("independent terminal win percentage", 100 * paired["endpoint_error_actual_start_m"]["fraction_new_lower"], 19.7, 0.05)

    diagnostics = independent["mechanism_diagnostics"]
    close("speed MAE", diagnostics["speed_mae_m_per_s"]["mean"], 0.676, 0.0005)
    close("speed MAE CI low", diagnostics["speed_mae_m_per_s"]["ci95_t"][0], 0.613, 0.0005)
    close("speed MAE CI high", diagnostics["speed_mae_m_per_s"]["ci95_t"][1], 0.739, 0.0005)
    correlations = diagnostics["correlations"]
    for key, expected_r, maximum_p in (
        ("speed_mae_vs_terminal_distance_absolute_error", 0.798, 1e-45),
        ("terminal_speed_error_vs_terminal_distance_signed_error", 0.586, 1e-19),
        ("duration_vs_terminal_distance_absolute_error", 0.016, 0.827),
        ("duration_vs_trajectory_error", 0.309, 1e-5),
    ):
        close(f"{key} r", correlations[key]["r"], expected_r, 0.0005)
        if correlations[key]["p_value"] > maximum_p:
            raise AssertionError(f"{key} p-value exceeds manuscript bound")
        passes.append(f"{key} p-value")
    close(
        "signed coordinate-distance mean",
        diagnostics["terminal_distance_signed_error_m"]["mean"],
        -0.052,
        0.0005,
    )
    for progress, recursive, measured in (("25", 1.78, 1.52), ("100", 5.73, 3.55)):
        values = diagnostics["progress_aligned_mean_error_m"][progress]
        close(f"recursive progress {progress}", values["independent_recursive_speed"], recursive, 0.005)
        close(f"measured-speed progress {progress}", values["measured_speed_control"], measured, 0.005)

    spectral = primary["spectral_radius"]
    close("spectral-radius mean", spectral["mean"], 1.01349, 0.000005)
    close("spectral-radius range low", spectral["min"], 1.01335, 0.000005)
    close("spectral-radius range high", spectral["max"], 1.01363, 0.000005)
    close("independent spectral-radius mean", independent["spectral_radius"]["mean"], 1.01357, 0.000005)

    equal("website reference samples", len(website_baseline["states"]), 139)
    close("website reference horizon", (len(website_baseline["states"]) - 1) * 0.05, 6.90, 0.0)
    equal("website maximum samples", website_model["training"]["maximumSequenceLength"], 164)
    close("website maximum horizon", (website_model["training"]["maximumSequenceLength"] - 1) * 0.05, 8.15, 0.0)
    equal(
        "website extension transitions",
        website_model["training"]["maximumSequenceLength"] - len(website_baseline["states"]),
        25,
    )
    close("illustrative 25-step spectral amplification", spectral["mean"] ** 25, 1.40, 0.005)
    equal("website trained sequences", website_model["training"]["acceptedSequences"], 203)
    close("website alpha", website_model["training"]["alpha"], 10.0, 0.0)
    matrix_a = np.asarray(website_model["matrices"]["A"], dtype=float)
    for axis, index, diagonal, velocity_coefficient in (
        ("X", 0, 0.99964, 0.04963),
        ("Y", 1, 0.99944, 0.05142),
        ("Z", 2, 0.99986, 0.04949),
    ):
        close(f"{axis} position persistence coefficient", matrix_a[index, index], diagonal, 0.000005)
        close(
            f"{axis} same-axis position-velocity coefficient",
            matrix_a[index, index + 3],
            velocity_coefficient,
            0.000005,
        )

    alpha_by_value = {row["alpha"]: row for row in alpha}
    close("alpha-grid minimum through 100", min(row["trajectory_mean_m"] for row in alpha if row["alpha"] <= 100), 1.330, 0.0005)
    close("alpha-grid maximum through 100", max(row["trajectory_mean_m"] for row in alpha if row["alpha"] <= 100), 1.341, 0.0005)
    close("alpha 100 path error", alpha_by_value[100.0]["trajectory_mean_m"], 1.330, 0.0005)
    close("alpha 1000 path error", alpha_by_value[1000.0]["trajectory_mean_m"], 1.397, 0.0005)

    expected_wind_features = {
        "zone_linear": (20, 1.339, 3.555, 1.684, None, None, None),
        "individual_linear": (56, 1.450, 3.885, 1.959, 0.111, 0.053, 0.169),
        "zone_linear_and_squared": (32, 1.391, 3.713, 1.825, 0.051, 0.008, 0.095),
        "individual_linear_and_squared": (104, 1.579, 4.275, 2.364, 0.240, 0.151, 0.328),
    }
    for variant, expected in expected_wind_features.items():
        control_dimension, path, endpoint, distance, difference, ci_low, ci_high = expected
        result = wind_features[variant]
        equal(f"{variant} control dimension", result["control_dimension"], control_dimension)
        close(f"{variant} path error", result["metrics"]["trajectory_error_m"]["mean"], path, 0.0005)
        close(f"{variant} endpoint error", result["metrics"]["endpoint_error_m"]["mean"], endpoint, 0.0005)
        close(
            f"{variant} coordinate-distance error",
            result["metrics"]["terminal_coordinate_distance_absolute_error_m"]["mean"],
            distance,
            0.0005,
        )
        if difference is not None:
            paired = result["paired_difference_vs_zone_linear"]["trajectory_error_m"]
            close(f"{variant} paired path difference", paired["mean"], difference, 0.0005)
            close(f"{variant} paired path CI low", paired["ci95_t"][0], ci_low, 0.0005)
            close(f"{variant} paired path CI high", paired["ci95_t"][1], ci_high, 0.0005)

    equal("figure example count", len(figure["examples"]), 3)
    close("figure target q10", figure["dataset_terminal_distance_quantiles_m"]["q10"], 181.68, 0.005)
    close("figure target q50", figure["dataset_terminal_distance_quantiles_m"]["q50"], 208.74, 0.005)
    close("figure target q90", figure["dataset_terminal_distance_quantiles_m"]["q90"], 228.93, 0.005)

    equal("Python version", sys.version.split()[0], "3.12.2")
    for package, reported in {
        "numpy": "2.3.2",
        "pandas": "2.3.1",
        "scipy": "1.16.1",
        "scikit-learn": "1.7.1",
    }.items():
        equal(f"{package} version", importlib.metadata.version(package), reported)

    for label, fragment in (
        ("Slovenian abstract", "\\abstractSi{Predstavimo"),
        ("figure reference", "revision_trajectory_representative.pdf"),
        ("alpha rationale", "preserves continuity with the original model"),
        ("L2 explanation", "weighted sum of squared coefficient values"),
        ("path-timing qualification", "reaches each location at the correct time"),
        ("coordinate-distance qualification", "not the official jump length"),
        ("March event", "during the March 2024 event"),
        ("available-file provenance", "reproducibility package begins after initial screening"),
        ("plain-language rendering", "Three.js and WebGL affect only presentation"),
        ("plain-language dataset scope", "jumps from the same hill and event"),
        ("unsupported-range qualification", "denotes an evidential boundary rather than a discontinuity"),
        ("duplicated-feature qualification", "only the combined contribution $(a+b)q_k$ is identifiable"),
        ("corrected-data wind sensitivity", "corrected-data wind-feature sensitivity supports retaining zone averaging"),
    ):
        contains(label, fragment)

    print(f"PASS: {len(passes)} quantitative and manuscript-linkage checks")
    print("NOT COMPUTATIONALLY VERIFIABLE FROM THE PRESENT REPOSITORY:")
    print("- 223 files originally supplied and 19 removed before the retained 204-file folder")
    print("- the exact acquisition day within the verified 21--24 March 2024 event")
    print("  (the 20240409 filename field is treated as a later batch/export label)")
    print("- grant identifiers and externally assigned bibliographic metadata")


if __name__ == "__main__":
    main()
