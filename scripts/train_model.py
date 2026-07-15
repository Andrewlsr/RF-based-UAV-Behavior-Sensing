"""
Train and package the AIR flight-mode prediction system.

This script is the deployment-oriented training entry point. It keeps the
research scripts unchanged while using the same validated RF feature
definitions:

    .dat -> IQ loading -> 20 ms segmentation -> RF features -> scaler
    -> Random Forest flight-mode classifier

The script also reruns recording-grouped validation before fitting the final
deployment model on all available AIR feature rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import FEATURE_NAMES, FeatureConfig, build_air_feature_rows
from src.iq_loader import DEFAULT_DATASET_ROOT, FLIGHT_MODE_LABELS
from src.model import (
    FLIGHT_MODE_CLASSES,
    calculate_metrics,
    recording_level_predictions,
    run_recording_grouped_validation,
    save_model_artifacts,
    train_final_flight_mode_model,
)
from src.preprocessing import fit_scaler, scale_features
from src.model import create_flight_mode_classifier


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "engineering_prediction_system"
DEFAULT_ANOVA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage5_factorial_information_analysis"
    / "robust_feature_evidence.csv"
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """
    Write dictionaries to CSV with a stable column order.

    Args:
        path: Destination CSV path.
        rows: Rows to write.
        fieldnames: Ordered CSV columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def progress_message(
    file_index: int,
    file_count: int,
    file_path: Path,
    file_rows: list[dict[str, object]],
) -> None:
    """
    Print compact per-recording feature extraction progress.

    Args:
        file_index: One-based index of the current file.
        file_count: Total number of files being processed.
        file_path: Current `.dat` path.
        file_rows: Segment feature rows extracted from the file.
    """
    state = str(file_rows[0]["state"]) if file_rows else "unknown"
    mode = str(file_rows[0]["mode_code"]) if file_rows else "unknown"
    peak_mean = np.mean([float(row["peak_frequency_mhz"]) for row in file_rows])
    entropy_mean = np.mean([float(row["spectral_entropy"]) for row in file_rows])
    print(
        f"[{file_index:02d}/{file_count:02d}] {state} {mode} {file_path.name} "
        f"segments={len(file_rows)} peak_mean={peak_mean:.2f} MHz "
        f"entropy_mean={entropy_mean:.3f}"
    )


def read_anova_comparison(anova_csv: Path) -> dict[str, dict[str, str]]:
    """
    Read Stage 5 factorial-analysis evidence if available.

    Args:
        anova_csv: Path to `robust_feature_evidence.csv`.

    Returns:
        Mapping from feature name to ANOVA evidence row. Returns an empty
        mapping when the file is absent.
    """
    if not anova_csv.exists():
        return {}
    with anova_csv.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    return {row["feature"]: row for row in rows}


def grouped_permutation_importance(
    rows: list[dict[str, object]],
    feature_names: list[str],
    repeats: int,
) -> dict[str, tuple[float, float]]:
    """
    Estimate feature importance by permuting held-out grouped folds.

    Args:
        rows: Segment-level feature rows.
        feature_names: Ordered model feature names.
        repeats: Number of shuffles per feature per fold.

    Returns:
        Mapping from feature name to `(mean_accuracy_drop, std_accuracy_drop)`.

    Notes:
        If shuffling one feature on held-out recordings reduces flight-mode
        accuracy, that feature is carrying behaviour information useful to the
        predictor.
    """
    from src.feature_extraction import feature_rows_to_matrix

    feature_matrix = feature_rows_to_matrix(rows, feature_names)
    labels = np.array([str(row["mode_code"]) for row in rows])
    fold_ids = sorted({str(row["recording_index"]) for row in rows})
    rng = np.random.default_rng(42)
    drops_by_feature = {feature_name: [] for feature_name in feature_names}

    for fold_id in fold_ids:
        train_indices = [
            index for index, row in enumerate(rows) if str(row["recording_index"]) != fold_id
        ]
        test_indices = [
            index for index, row in enumerate(rows) if str(row["recording_index"]) == fold_id
        ]
        scaler = fit_scaler(feature_matrix[train_indices])
        classifier = create_flight_mode_classifier()
        classifier.fit(scale_features(feature_matrix[train_indices], scaler), labels[train_indices])

        x_test = feature_matrix[test_indices]
        test_rows = [rows[index] for index in test_indices]
        baseline_predictions = [
            str(label) for label in classifier.predict(scale_features(x_test, scaler))
        ]
        _, baseline_true, baseline_recording_pred = recording_level_predictions(
            test_rows,
            baseline_predictions,
            target_column="mode_code",
            class_order=FLIGHT_MODE_CLASSES,
        )
        baseline_score = calculate_metrics(baseline_true, baseline_recording_pred)[
            "balanced_accuracy"
        ]

        for feature_index, feature_name in enumerate(feature_names):
            for _ in range(repeats):
                x_permuted = np.array(x_test, copy=True)
                x_permuted[:, feature_index] = rng.permutation(x_permuted[:, feature_index])
                permuted_predictions = [
                    str(label)
                    for label in classifier.predict(scale_features(x_permuted, scaler))
                ]
                _, permuted_true, permuted_recording_pred = recording_level_predictions(
                    test_rows,
                    permuted_predictions,
                    target_column="mode_code",
                    class_order=FLIGHT_MODE_CLASSES,
                )
                permuted_score = calculate_metrics(
                    permuted_true,
                    permuted_recording_pred,
                )["balanced_accuracy"]
                drops_by_feature[feature_name].append(baseline_score - permuted_score)

    return {
        feature_name: (
            float(np.mean(drops)),
            float(np.std(drops, ddof=0)),
        )
        for feature_name, drops in drops_by_feature.items()
    }


def make_feature_importance_rows(
    classifier,
    rows: list[dict[str, object]],
    repeats: int,
    anova_csv: Path,
) -> list[dict[str, object]]:
    """
    Combine Random Forest, permutation, and Stage 5 ANOVA evidence.

    Args:
        classifier: Trained final Random Forest.
        rows: Segment-level feature rows used for validation/explanation.
        repeats: Number of grouped permutation repeats.
        anova_csv: Optional Stage 5 evidence table.

    Returns:
        Ranked feature-importance rows.
    """
    permutation = grouped_permutation_importance(rows, FEATURE_NAMES, repeats)
    anova = read_anova_comparison(anova_csv)

    importance_rows = []
    for feature_name, rf_importance in zip(FEATURE_NAMES, classifier.feature_importances_):
        anova_row = anova.get(feature_name, {})
        permutation_mean, permutation_std = permutation[feature_name]
        importance_rows.append(
            {
                "feature": feature_name,
                "random_forest_importance": float(rf_importance),
                "grouped_permutation_importance_mean": permutation_mean,
                "grouped_permutation_importance_std": permutation_std,
                "stage5_robust_effects": anova_row.get("robust_effects", ""),
                "stage5_strongest_effect": anova_row.get("strongest_robust_effect", ""),
                "stage5_engineering_conclusion": anova_row.get(
                    "engineering_conclusion",
                    "",
                ),
            }
        )

    importance_rows.sort(
        key=lambda row: (
            float(row["grouped_permutation_importance_mean"]),
            float(row["random_forest_importance"]),
        ),
        reverse=True,
    )
    return importance_rows


def plot_feature_importance(rows: list[dict[str, object]], output_path: Path) -> None:
    """
    Plot Random Forest and grouped permutation importance.

    Args:
        rows: Ranked feature-importance rows.
        output_path: Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = list(reversed(rows))
    features = [str(row["feature"]) for row in ordered_rows]
    rf_values = [float(row["random_forest_importance"]) for row in ordered_rows]
    perm_values = [
        float(row["grouped_permutation_importance_mean"]) for row in ordered_rows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = np.arange(len(features))

    axes[0].barh(y, rf_values, color="tab:blue")
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].set_xlabel("Mean impurity decrease")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(features)

    axes[1].barh(y, perm_values, color="tab:green")
    axes[1].set_title("Grouped Permutation Importance")
    axes[1].set_xlabel("Drop in recording balanced accuracy")

    figure.suptitle("Why Flight Mode Can Be Predicted from RF Features")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line training options.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the deployable AIR ON/HO/FY RF behaviour predictor."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--segments-per-recording",
        type=int,
        default=10,
        help=(
            "20 ms segments per recording. Default 10 reproduces the Stage 6 "
            "validated setting. Use 0 for all non-overlapping 20 ms segments."
        ),
    )
    parser.add_argument("--segment-ms", type=float, default=20.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--anova-csv", type=Path, default=DEFAULT_ANOVA_CSV)
    return parser.parse_args()


def main() -> None:
    """
    Train, validate, explain, and save the engineering predictor.

    Returns:
        None.

    Notes:
        The grouped-validation result printed by this function is the
        performance estimate. The final model is then trained on all available
        AIR rows for deployment.
    """
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(
        segment_ms=args.segment_ms,
        segments_per_recording=args.segments_per_recording,
    )

    print("\nRF UAV Behaviour Predictor - Training")
    print(f"Dataset root: {args.data_root}")
    print(f"Segment length: {config.segment_ms:.1f} ms")
    if config.segments_per_recording <= 0:
        print("Segments per recording: all non-overlapping 20 ms segments")
    else:
        print(f"Segments per recording: {config.segments_per_recording}")
    print("\nExtracting RF features from AIR .dat recordings...")

    rows = build_air_feature_rows(
        dataset_root=args.data_root,
        config=config,
        max_files=args.max_files,
        progress_callback=progress_message,
    )

    metadata_fields = [
        "file_path",
        "file_name",
        "model",
        "interference_code",
        "interference_label",
        "mode_code",
        "mode_label",
        "state",
        "recording_index",
        "segment_index",
        "segments_per_recording",
        "segment_start_sample",
        "segment_start_ms",
        "segment_duration_ms",
        "samples_read",
    ]
    feature_matrix_csv = args.output_dir / "training_feature_matrix.csv"
    write_csv(feature_matrix_csv, rows, metadata_fields + FEATURE_NAMES)

    print("\nRunning recording-grouped validation...")
    fold_rows, validation_summary, validation_predictions = run_recording_grouped_validation(
        rows,
        FEATURE_NAMES,
    )
    write_csv(
        args.output_dir / "grouped_validation_folds.csv",
        fold_rows,
        [
            "fold",
            "test_recordings",
            "test_segments",
            "segment_balanced_accuracy",
            "segment_macro_f1",
            "recording_balanced_accuracy",
            "recording_macro_f1",
        ],
    )
    write_csv(
        args.output_dir / "grouped_validation_predictions.csv",
        validation_predictions,
        ["fold", "file_path", "segment_index", "true_mode", "predicted_mode"],
    )
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2),
        encoding="utf-8",
    )

    print("\nTraining final Random Forest on all AIR feature rows...")
    classifier, scaler, _, _ = train_final_flight_mode_model(rows, FEATURE_NAMES)
    save_model_artifacts(
        classifier=classifier,
        scaler=scaler,
        config=config,
        model_dir=args.model_dir,
        validation_summary=validation_summary,
        training_rows=rows,
        feature_names=FEATURE_NAMES,
    )

    print("\nGenerating explainability outputs...")
    importance_rows = make_feature_importance_rows(
        classifier,
        rows,
        repeats=args.permutation_repeats,
        anova_csv=args.anova_csv,
    )
    write_csv(
        args.output_dir / "feature_importance.csv",
        importance_rows,
        [
            "feature",
            "random_forest_importance",
            "grouped_permutation_importance_mean",
            "grouped_permutation_importance_std",
            "stage5_robust_effects",
            "stage5_strongest_effect",
            "stage5_engineering_conclusion",
        ],
    )
    plot_feature_importance(importance_rows, args.output_dir / "feature_importance.png")

    print("\nTraining complete.")
    print(f"Feature rows: {len(rows)}")
    print(f"Source recordings: {len({str(row['file_path']) for row in rows})}")
    print(
        "Recording-level grouped validation balanced accuracy: "
        f"{validation_summary['recording_balanced_accuracy']:.3f}"
    )
    print(f"Saved classifier: {(args.model_dir / 'classifier.joblib').resolve()}")
    print(f"Saved scaler: {(args.model_dir / 'scaler.joblib').resolve()}")
    print(f"Saved feature config: {(args.model_dir / 'feature_config.json').resolve()}")
    print(f"Saved feature importance: {(args.output_dir / 'feature_importance.csv').resolve()}")
    print("\nClass labels:")
    for label in FLIGHT_MODE_CLASSES:
        print(f"  {label}: {FLIGHT_MODE_LABELS[label]}")


if __name__ == "__main__":
    main()
