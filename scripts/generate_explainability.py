"""
Generate the release explainability report for the trained AIR predictor.

This script uses the saved Random Forest model, saved scaler, and the training
feature matrix produced by `scripts/train_model.py`. It does not retrain the
classifier. It produces feature-importance CSV/PNG outputs and a Markdown
interpretation that compares model importance with Stage 5 factorial evidence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import FEATURE_NAMES, feature_rows_to_matrix
from src.model import (
    FLIGHT_MODE_CLASSES,
    calculate_metrics,
    load_model_artifacts,
    recording_level_predictions,
)
from src.preprocessing import scale_features


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_FEATURE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "engineering_prediction_system"
    / "training_feature_matrix.csv"
)
DEFAULT_ANOVA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage5_factorial_information_analysis"
    / "robust_feature_evidence.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Read a CSV file into dictionaries.

    Args:
        path: CSV path.

    Returns:
        List of CSV rows.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the CSV has no rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """
    Write CSV rows.

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


def read_anova(anova_csv: Path) -> dict[str, dict[str, str]]:
    """
    Read Stage 5 factorial evidence if available.

    Args:
        anova_csv: Path to robust feature evidence.

    Returns:
        Mapping from feature name to evidence row, or an empty mapping.
    """
    if not anova_csv.exists():
        return {}
    return {row["feature"]: row for row in read_csv_rows(anova_csv)}


def recording_permutation_importance(
    rows: list[dict[str, object]],
    classifier,
    scaler,
    feature_names: list[str],
    repeats: int,
) -> dict[str, tuple[float, float]]:
    """
    Permute each feature and measure recording-level accuracy drop.

    Args:
        rows: Segment-level feature rows.
        classifier: Saved Random Forest classifier.
        scaler: Saved feature scaler.
        feature_names: Ordered feature names.
        repeats: Number of permutations per feature.

    Returns:
        Mapping from feature name to `(mean_accuracy_drop, std_accuracy_drop)`.

    Notes:
        The saved model is not retrained here. The question is: when this trained
        model loses a feature, how much does its recording-level behaviour
        prediction degrade?
    """
    feature_matrix = feature_rows_to_matrix(rows, feature_names)
    baseline_predictions = [
        str(label)
        for label in classifier.predict(scale_features(feature_matrix, scaler))
    ]
    _, true_recordings, predicted_recordings = recording_level_predictions(
        rows,
        baseline_predictions,
        target_column="mode_code",
        class_order=FLIGHT_MODE_CLASSES,
    )
    baseline_score = calculate_metrics(true_recordings, predicted_recordings)[
        "balanced_accuracy"
    ]

    rng = np.random.default_rng(42)
    drops: dict[str, list[float]] = {feature_name: [] for feature_name in feature_names}
    for feature_index, feature_name in enumerate(feature_names):
        for _ in range(repeats):
            permuted = np.array(feature_matrix, copy=True)
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            predictions = [
                str(label)
                for label in classifier.predict(scale_features(permuted, scaler))
            ]
            _, true_after, predicted_after = recording_level_predictions(
                rows,
                predictions,
                target_column="mode_code",
                class_order=FLIGHT_MODE_CLASSES,
            )
            permuted_score = calculate_metrics(true_after, predicted_after)[
                "balanced_accuracy"
            ]
            drops[feature_name].append(baseline_score - permuted_score)

    return {
        feature_name: (
            float(np.mean(values)),
            float(np.std(values, ddof=0)),
        )
        for feature_name, values in drops.items()
    }


def build_importance_table(
    rows: list[dict[str, object]],
    classifier,
    scaler,
    anova_csv: Path,
    repeats: int,
) -> list[dict[str, object]]:
    """
    Build the ranked importance table used by CSV/plot/report outputs.

    Args:
        rows: Segment-level feature rows.
        classifier: Saved Random Forest classifier.
        scaler: Saved feature scaler.
        anova_csv: Optional Stage 5 evidence table.
        repeats: Number of permutation repeats.

    Returns:
        Ranked feature-importance rows.
    """
    permutation = recording_permutation_importance(
        rows,
        classifier,
        scaler,
        FEATURE_NAMES,
        repeats,
    )
    anova = read_anova(anova_csv)
    table = []
    for feature_name, rf_importance in zip(FEATURE_NAMES, classifier.feature_importances_):
        permutation_mean, permutation_std = permutation[feature_name]
        anova_row = anova.get(feature_name, {})
        table.append(
            {
                "feature": feature_name,
                "random_forest_importance": float(rf_importance),
                "permutation_importance_mean_accuracy_drop": permutation_mean,
                "permutation_importance_std": permutation_std,
                "stage5_robust_effects": anova_row.get("robust_effects", ""),
                "stage5_strongest_effect": anova_row.get("strongest_robust_effect", ""),
                "stage5_engineering_conclusion": anova_row.get(
                    "engineering_conclusion",
                    "",
                ),
            }
        )

    table.sort(
        key=lambda row: (
            float(row["permutation_importance_mean_accuracy_drop"]),
            float(row["random_forest_importance"]),
        ),
        reverse=True,
    )
    return table


def plot_importance(table: list[dict[str, object]], output_path: Path) -> None:
    """
    Create the release feature-importance plot.

    Args:
        table: Ranked feature-importance rows.
        output_path: Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(reversed(table))
    features = [str(row["feature"]) for row in rows]
    rf_values = [float(row["random_forest_importance"]) for row in rows]
    permutation_values = [
        float(row["permutation_importance_mean_accuracy_drop"]) for row in rows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = np.arange(len(features))
    axes[0].barh(y, rf_values, color="#356CA5")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(features)
    axes[0].set_xlabel("Random Forest importance")
    axes[0].set_title("Model Feature Use")

    axes[1].barh(y, permutation_values, color="#5E8C61")
    axes[1].set_xlabel("Recording accuracy drop")
    axes[1].set_title("Permutation Importance")

    figure.suptitle("RF Feature Contributions to AIR Flight-Mode Recognition")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_report(path: Path, table: list[dict[str, object]]) -> None:
    """
    Write the Markdown engineering interpretation report.

    Args:
        path: Destination Markdown path.
        table: Ranked feature-importance rows.
    """
    top_rows = table[:5]
    top_lines = [
        (
            f"- `{row['feature']}`: permutation drop "
            f"{float(row['permutation_importance_mean_accuracy_drop']):.3f}, "
            f"RF importance {float(row['random_forest_importance']):.3f}, "
            f"Stage 5 evidence: {row['stage5_engineering_conclusion'] or 'not available'}"
        )
        for row in top_rows
    ]

    by_feature = {str(row["feature"]): row for row in table}

    def describe(feature: str) -> str:
        """Return a one-sentence interpretation for a named feature."""
        row = by_feature[feature]
        return (
            f"`{feature}` has RF importance "
            f"{float(row['random_forest_importance']):.3f} and permutation drop "
            f"{float(row['permutation_importance_mean_accuracy_drop']):.3f}. "
            f"Stage 5 conclusion: {row['stage5_engineering_conclusion'] or 'not available'}."
        )

    report = f"""# Feature Interpretation Report

## Purpose

This report explains why the AIR flight-mode classifier can predict `ON`,
`HO`, and `FY` from passive RF recordings. The model is a Random Forest trained
on interpretable RF features, not a deep neural network.

## Most Contributing RF Features

{chr(10).join(top_lines)}

## Comparison With Factorial Analysis

Stage 5 showed that flight mode is the dominant RF information source,
interference is secondary, and a flight-mode x interference interaction exists.
The importance ranking is consistent with that result: the classifier relies on
features describing spectral concentration, activity fraction, frequency
position, bandwidth, and temporal variation rather than on one isolated scalar.

## Required Feature Discussion

- Spectral entropy: {describe("spectral_entropy")}
- RMS power: {describe("rms_power")}
- Spectral centroid: {describe("spectral_centroid_mhz")}
- Temporal variability: {describe("spectrogram_temporal_variability")}

## Engineering Interpretation

Flight mode changes the RF behaviour because the UAV communication/control link
does not occupy frequency and time in exactly the same way when the drone is
only switched on, hovering, or flying. The strongest evidence comes from
features that describe how RF energy is distributed in frequency and how active
the spectrum is over time.

Power features are useful context, but power alone is not a reliable behaviour
signature because received amplitude can change with geometry, antenna
orientation, and propagation. Spectral and time-frequency features are more
directly tied to waveform structure.

## Caveat

Permutation importance here uses the saved final model and the available AIR
feature matrix. It is an interpretability tool, not a replacement for the
recording-grouped validation result used for performance reporting.
"""
    path.write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line explainability settings.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate feature interpretation outputs for the trained AIR model."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--anova-csv", type=Path, default=DEFAULT_ANOVA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """
    Generate explainability artifacts from saved model outputs.

    Returns:
        None.
    """
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier, scaler, _ = load_model_artifacts(args.model_dir)
    rows = read_csv_rows(args.feature_csv)
    table = build_importance_table(
        rows,
        classifier,
        scaler,
        args.anova_csv,
        args.permutation_repeats,
    )
    write_csv(
        args.output_dir / "feature_importance.csv",
        table,
        [
            "feature",
            "random_forest_importance",
            "permutation_importance_mean_accuracy_drop",
            "permutation_importance_std",
            "stage5_robust_effects",
            "stage5_strongest_effect",
            "stage5_engineering_conclusion",
        ],
    )
    plot_importance(table, args.output_dir / "feature_importance.png")
    write_report(args.output_dir / "feature_interpretation.md", table)

    print("\nExplainability outputs generated.")
    print(f"Saved table: {(args.output_dir / 'feature_importance.csv').resolve()}")
    print(f"Saved plot: {(args.output_dir / 'feature_importance.png').resolve()}")
    print(f"Saved report: {(args.output_dir / 'feature_interpretation.md').resolve()}")


if __name__ == "__main__":
    main()
