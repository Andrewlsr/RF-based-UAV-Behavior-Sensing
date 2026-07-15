"""
Stage 7A: Train the final AIR flight-mode predictor.

Purpose:
    Build a deployable traditional-ML predictor for the strongest validated
    target:

        Flight mode = ON / HO / FY

Why flight mode first:
    Stage 6 leakage-safe validation showed that flight mode is the most
    reliable generalizable RF source:

        Random Forest recording-level balanced accuracy: about 80%
        Chance level: 33.3%

Important:
    This script trains the final model on all available AIR feature rows after
    validation has already been completed. It does not report a new unbiased
    accuracy estimate. The unbiased estimate comes from Stage 6.

Input:
    outputs/stage4_air_state_analysis/per_record_features.csv

Output:
    outputs/stage7_prediction_model/flight_mode_predictor.joblib
"""

from pathlib import Path

import argparse
import csv
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from stage4_air_state_analysis import (
    DEFAULT_SEGMENT_MS,
    DEFAULT_SEGMENTS_PER_FILE,
    DEFAULT_PSD_NPERSEG,
    DEFAULT_PSD_OVERLAP,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SPECTROGRAM_NPERSEG,
    DEFAULT_SPECTROGRAM_OVERLAP,
    DEFAULT_DC_EXCLUSION_HZ,
    FEATURE_NAMES,
)


DEFAULT_FEATURE_CSV = (
    Path("outputs")
    / "stage4_air_state_analysis"
    / "per_record_features.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage7_prediction_model"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "flight_mode_predictor.joblib"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "flight_mode_predictor_metadata.json"
RANDOM_STATE = 42


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    """Read the segment-level feature table produced by Stage 4."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {path}\n"
            "Run stage4_air_state_analysis.py first."
        )

    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))

    if not rows:
        raise ValueError(f"Feature matrix is empty: {path}")
    return rows


def build_feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    """Convert RF feature columns to the numeric training matrix."""
    matrix = np.array(
        [
            [float(row[feature_name]) for feature_name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return matrix


def train_flight_mode_model(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
) -> RandomForestClassifier:
    """
    Train the final Random Forest flight-mode classifier.

    This uses the same fixed model family that performed best in Stage 6.
    We avoid extra tuning here because proper tuning would need nested grouped
    validation.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(feature_matrix, labels)
    return model


def save_model_bundle(
    model: RandomForestClassifier,
    model_path: Path,
    metadata_path: Path,
    training_rows: list[dict[str, str]],
) -> None:
    """Save the model and a human-readable metadata sidecar."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    source_recordings = sorted({row["file_path"] for row in training_rows})
    class_counts = {
        label: int(sum(row["mode_code"] == label for row in training_rows))
        for label in model.classes_
    }
    recording_counts = {
        label: len(
            {
                row["file_path"]
                for row in training_rows
                if row["mode_code"] == label
            }
        )
        for label in model.classes_
    }

    bundle = {
        "model": model,
        "target": "flight_mode",
        "target_column": "mode_code",
        "class_labels": list(model.classes_),
        "feature_names": list(FEATURE_NAMES),
        "segment_ms": DEFAULT_SEGMENT_MS,
        "segments_per_file": DEFAULT_SEGMENTS_PER_FILE,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "psd_nperseg": DEFAULT_PSD_NPERSEG,
        "psd_overlap": DEFAULT_PSD_OVERLAP,
        "spectrogram_nperseg": DEFAULT_SPECTROGRAM_NPERSEG,
        "spectrogram_overlap": DEFAULT_SPECTROGRAM_OVERLAP,
        "dc_exclusion_hz": DEFAULT_DC_EXCLUSION_HZ,
        "validated_by": "stage6_leakage_safe_ml_validation.py",
        "validated_recording_level_balanced_accuracy": 0.80,
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "model_path": str(model_path),
        "target": "flight_mode",
        "classes": list(model.classes_),
        "feature_names": list(FEATURE_NAMES),
        "training_feature_rows": len(training_rows),
        "training_recordings": len(source_recordings),
        "segment_rows_per_class": class_counts,
        "source_recordings_per_class": recording_counts,
        "segment_ms": DEFAULT_SEGMENT_MS,
        "segments_per_file": DEFAULT_SEGMENTS_PER_FILE,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "validated_recording_level_balanced_accuracy": 0.80,
        "validation_note": (
            "Accuracy estimate comes from leakage-safe Stage 6 grouped "
            "validation. This final model is trained on all available AIR "
            "feature rows for deployment/prediction."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """Parse training settings."""
    parser = argparse.ArgumentParser(
        description="Train final AIR flight-mode predictor from extracted RF features."
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=DEFAULT_FEATURE_CSV,
        help="Stage 4 feature CSV. Default: outputs/stage4_air_state_analysis/per_record_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to save human-readable model metadata.",
    )
    return parser.parse_args()


def main() -> None:
    """Train and save the final flight-mode prediction model."""
    args = parse_arguments()
    rows = read_feature_rows(args.feature_csv)
    feature_matrix = build_feature_matrix(rows)
    labels = np.array([row["mode_code"] for row in rows])

    model = train_flight_mode_model(feature_matrix, labels)
    save_model_bundle(model, args.model_path, args.metadata_path, rows)

    print("\n--- Final AIR flight-mode predictor trained ---")
    print(f"Training feature rows: {len(rows)}")
    print(f"Source recordings: {len({row['file_path'] for row in rows})}")
    print(f"Classes: {', '.join(model.classes_)}")
    print(f"Saved model: {args.model_path.resolve()}")
    print(f"Saved metadata: {args.metadata_path.resolve()}")
    print(
        "Validation reference: Stage 6 leakage-safe recording-level balanced "
        "accuracy was about 80%."
    )


if __name__ == "__main__":
    main()
