"""
Train, validate, and persist traditional ML models for AIR flight mode.

This module owns the leakage-safe validation logic and final model training.
It deliberately uses traditional machine learning rather than deep learning:
the validated deployment model is a Random Forest over interpretable RF
features.

Notes:
    The most important validation rule is recording-level grouping. A single
    `.dat` recording contributes multiple 20 ms segments. If some segments from
    one recording were placed in training and other segments from the same
    recording were placed in testing, the validation result would be overly
    optimistic. Grouping by original recording prevents this segment-level
    leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.feature_extraction import FEATURE_NAMES, FeatureConfig, feature_rows_to_matrix
from src.iq_loader import FLIGHT_MODE_LABELS
from src.preprocessing import fit_scaler, scale_features


RANDOM_STATE = 42
FLIGHT_MODE_CLASSES = ["ON", "HO", "FY"]


def create_flight_mode_classifier() -> RandomForestClassifier:
    """
    Create the validated traditional ML classifier.

    Returns:
        Configured `RandomForestClassifier`.

    Notes:
        Random Forest is used because Stage 6 showed it gave the strongest
        leakage-safe flight-mode result while retaining feature importance tools.
    """
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def majority_vote(labels: Iterable[str], class_order: list[str]) -> str:
    """
    Return the most common class with deterministic tie-breaking.

    Args:
        labels: Segment-level predicted class labels.
        class_order: Stable class order used to break ties.

    Returns:
        Majority-vote label.

    Notes:
        Ties are resolved by the saved class order so repeated runs are
        deterministic.
    """
    label_list = list(labels)
    counts = {label: label_list.count(label) for label in class_order}
    return max(class_order, key=lambda label: (counts[label], -class_order.index(label)))


def recording_level_predictions(
    rows: list[dict[str, object]],
    segment_predictions: Iterable[str],
    target_column: str = "mode_code",
    class_order: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Combine segment predictions into one prediction per recording.

    Args:
        rows: Segment-level metadata rows.
        segment_predictions: Segment-level model predictions aligned with `rows`.
        target_column: Metadata field containing the true label.
        class_order: Stable class order used for majority voting.

    Returns:
        Tuple `(recording_paths, true_labels, predicted_labels)`.

    Raises:
        ValueError: If segments from one recording do not share one true label.

    Notes:
        Recording-level prediction is the engineering unit of interest because
        deployment receives one `.dat` recording and should return one flight
        mode decision.
    """
    class_order = class_order or FLIGHT_MODE_CLASSES
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    predictions = list(segment_predictions)
    for index, row in enumerate(rows):
        grouped_indices[str(row["file_path"])].append(index)

    recording_paths: list[str] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []

    for file_path, indices in sorted(grouped_indices.items()):
        true_values = {str(rows[index][target_column]) for index in indices}
        if len(true_values) != 1:
            raise ValueError(f"Recording contains inconsistent labels: {file_path}")

        recording_paths.append(file_path)
        true_labels.append(true_values.pop())
        predicted_labels.append(majority_vote([predictions[index] for index in indices], class_order))

    return recording_paths, true_labels, predicted_labels


def calculate_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict[str, float]:
    """
    Calculate class-balanced validation metrics.

    Args:
        true_labels: Ground-truth labels.
        predicted_labels: Predicted labels.

    Returns:
        Dictionary with balanced accuracy and macro F1.

    Notes:
        Balanced accuracy is used because it treats each class equally even if
        class counts change in later experiments.
    """
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
    }


def validate_grouped_dataset(rows: list[dict[str, object]]) -> list[str]:
    """
    Validate recording-group structure and return fold IDs.

    Args:
        rows: Segment-level feature rows containing `file_path` and
            `recording_index`.

    Returns:
        Sorted recording indices used as grouped validation folds.

    Raises:
        ValueError: If the dataset has too few recordings or fold IDs.

    Notes:
        The expected DroneDetect AIR structure has five recording indices per
        state. Fold 00 tests all recordings with index 00, fold 01 tests all index
        01 recordings, and so on. This prevents segment leakage.
    """
    recording_paths = sorted({str(row["file_path"]) for row in rows})
    fold_ids = sorted({str(row["recording_index"]) for row in rows})
    if len(recording_paths) < 2:
        raise ValueError("Grouped validation needs at least two recordings.")
    if len(fold_ids) < 2:
        raise ValueError("Grouped validation needs at least two recording indices.")
    return fold_ids


def run_recording_grouped_validation(
    rows: list[dict[str, object]],
    feature_names: list[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """
    Run leakage-safe grouped validation for flight-mode prediction.

    Args:
        rows: Segment-level RF feature rows.
        feature_names: Ordered feature columns used for model input.

    Returns:
        Tuple containing fold metrics, overall metrics, and segment prediction
        rows.

    Raises:
        RuntimeError: If any original recording appears in both train and test.

    Notes:
        Scaling and model fitting are performed inside each fold. No segment from a
        held-out recording is allowed to appear in training.
    """
    feature_names = feature_names or FEATURE_NAMES
    feature_matrix = feature_rows_to_matrix(rows, feature_names)
    labels = np.array([str(row["mode_code"]) for row in rows])
    fold_ids = validate_grouped_dataset(rows)

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    pooled_segment_true: list[str] = []
    pooled_segment_predicted: list[str] = []
    pooled_recording_true: list[str] = []
    pooled_recording_predicted: list[str] = []

    for fold_id in fold_ids:
        train_indices = [
            index for index, row in enumerate(rows) if str(row["recording_index"]) != fold_id
        ]
        test_indices = [
            index for index, row in enumerate(rows) if str(row["recording_index"]) == fold_id
        ]

        train_recordings = {str(rows[index]["file_path"]) for index in train_indices}
        test_recordings = {str(rows[index]["file_path"]) for index in test_indices}
        if train_recordings & test_recordings:
            raise RuntimeError("Recording leakage detected between train and test.")

        scaler = fit_scaler(feature_matrix[train_indices])
        classifier = create_flight_mode_classifier()
        classifier.fit(scale_features(feature_matrix[train_indices], scaler), labels[train_indices])
        segment_predictions = [
            str(label)
            for label in classifier.predict(scale_features(feature_matrix[test_indices], scaler))
        ]
        segment_true = [str(labels[index]) for index in test_indices]

        test_rows = [rows[index] for index in test_indices]
        _, recording_true, recording_predicted = recording_level_predictions(
            test_rows,
            segment_predictions,
            target_column="mode_code",
            class_order=FLIGHT_MODE_CLASSES,
        )

        segment_metrics = calculate_metrics(segment_true, segment_predictions)
        recording_metrics = calculate_metrics(recording_true, recording_predicted)

        fold_rows.append(
            {
                "fold": fold_id,
                "test_recordings": len(test_recordings),
                "test_segments": len(test_indices),
                "segment_balanced_accuracy": segment_metrics["balanced_accuracy"],
                "segment_macro_f1": segment_metrics["macro_f1"],
                "recording_balanced_accuracy": recording_metrics["balanced_accuracy"],
                "recording_macro_f1": recording_metrics["macro_f1"],
            }
        )

        pooled_segment_true.extend(segment_true)
        pooled_segment_predicted.extend(segment_predictions)
        pooled_recording_true.extend(recording_true)
        pooled_recording_predicted.extend(recording_predicted)

        for row, true_label, predicted_label in zip(test_rows, segment_true, segment_predictions):
            prediction_rows.append(
                {
                    "fold": fold_id,
                    "file_path": row["file_path"],
                    "segment_index": row["segment_index"],
                    "true_mode": true_label,
                    "predicted_mode": predicted_label,
                }
            )

    segment_overall = calculate_metrics(pooled_segment_true, pooled_segment_predicted)
    recording_overall = calculate_metrics(pooled_recording_true, pooled_recording_predicted)
    overall = {
        "fold_count": len(fold_ids),
        "segment_count": len(pooled_segment_true),
        "recording_count": len(pooled_recording_true),
        "segment_balanced_accuracy": segment_overall["balanced_accuracy"],
        "segment_macro_f1": segment_overall["macro_f1"],
        "recording_balanced_accuracy": recording_overall["balanced_accuracy"],
        "recording_macro_f1": recording_overall["macro_f1"],
        "chance_balanced_accuracy": 1.0 / len(FLIGHT_MODE_CLASSES),
    }
    return fold_rows, overall, prediction_rows


def train_final_flight_mode_model(
    rows: list[dict[str, object]],
    feature_names: list[str] | None = None,
) -> tuple[RandomForestClassifier, object, np.ndarray, np.ndarray]:
    """
    Fit the final scaler and Random Forest on all available training rows.

    Args:
        rows: Segment-level feature rows.
        feature_names: Ordered feature columns used for model input.

    Returns:
        Tuple `(classifier, scaler, feature_matrix, labels)`.

    Notes:
        This is the deployment model fit after leakage-safe validation has
        already estimated performance. The unbiased performance estimate should
        still be taken from grouped validation, not from this final fit.
    """
    feature_names = feature_names or FEATURE_NAMES
    feature_matrix = feature_rows_to_matrix(rows, feature_names)
    labels = np.array([str(row["mode_code"]) for row in rows])
    scaler = fit_scaler(feature_matrix)
    classifier = create_flight_mode_classifier()
    classifier.fit(scale_features(feature_matrix, scaler), labels)
    return classifier, scaler, feature_matrix, labels


def save_model_artifacts(
    classifier: RandomForestClassifier,
    scaler: object,
    config: FeatureConfig,
    model_dir: Path,
    validation_summary: dict[str, object],
    training_rows: list[dict[str, object]],
    feature_names: list[str] | None = None,
) -> None:
    """
    Save classifier, scaler, and feature configuration for deployment.

    Args:
        classifier: Trained Random Forest model.
        scaler: Fitted feature scaler.
        config: Feature extraction configuration.
        model_dir: Destination folder for model artifacts.
        validation_summary: Grouped-validation summary to store with metadata.
        training_rows: Segment-level rows used for final training.
        feature_names: Ordered feature columns expected by the model.

    Returns:
        None.

    Notes:
        `feature_config.json` is as important as the model itself: it preserves
        segment length, sample rate, feature order, class order, and validation
        summary for reproducible prediction.
    """
    feature_names = feature_names or FEATURE_NAMES
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, model_dir / "classifier.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    class_counts = Counter(str(row["mode_code"]) for row in training_rows)
    recording_counts = {
        label: len(
            {
                str(row["file_path"])
                for row in training_rows
                if str(row["mode_code"]) == label
            }
        )
        for label in FLIGHT_MODE_CLASSES
    }

    feature_config = {
        **config.to_dict(),
        "target": "flight_mode",
        "target_column": "mode_code",
        "feature_names": feature_names,
        "class_order": FLIGHT_MODE_CLASSES,
        "label_mapping": FLIGHT_MODE_LABELS,
        "segment_rows_per_class": dict(class_counts),
        "source_recordings_per_class": recording_counts,
        "training_feature_rows": len(training_rows),
        "training_recordings": len({str(row["file_path"]) for row in training_rows}),
        "validation": validation_summary,
        "rf_meaning": (
            "The classifier uses power, spectral position, occupied bandwidth, "
            "spectral entropy, and spectrogram activity features extracted from "
            "passively received AIR IQ recordings."
        ),
    }

    import json

    (model_dir / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2),
        encoding="utf-8",
    )


def load_model_artifacts(model_dir: Path) -> tuple[RandomForestClassifier, object, dict[str, object]]:
    """
    Load saved classifier, scaler, and feature configuration.

    Args:
        model_dir: Folder containing `classifier.joblib`, `scaler.joblib`, and
            `feature_config.json`.

    Returns:
        Tuple `(classifier, scaler, feature_config)`.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    import json

    classifier_path = model_dir / "classifier.joblib"
    scaler_path = model_dir / "scaler.joblib"
    config_path = model_dir / "feature_config.json"

    missing = [path for path in (classifier_path, scaler_path, config_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model artifact(s):\n{missing_text}")

    classifier = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return classifier, scaler, config
