"""
Deployment prediction pipeline for AIR UAV flight mode.

Architecture:
    Original production RF
        -> baseline prediction
    Additional 40 ms temporal RF
        -> validated FY/HO temporal optimization
        -> final prediction

The original production model remains intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.feature_extraction import (
    FEATURE_NAMES,
    FeatureConfig,
    extract_recording_feature_rows,
    feature_rows_to_matrix,
)
from src.iq_loader import FLIGHT_MODE_LABELS
from src.model import FLIGHT_MODE_CLASSES, load_model_artifacts, majority_vote
from src.preprocessing import scale_features
from src.temporal_optimization import optimize_prediction


@dataclass(frozen=True)
class PredictionResult:
    file_path: Path
    final_prediction: str
    final_prediction_label: str
    confidence: float
    vote_counts: dict[str, int]
    mean_probabilities: dict[str, float]
    feature_rows: list[dict[str, object]]
    segment_predictions: list[str]

    # New diagnostic fields; baseline remains available.
    baseline_prediction: str
    optimized_prediction: str
    optimization_applied: bool
    optimization_reason: str


def _predict_original(
    file_path: Path,
    model_dir: Path,
    segments_per_recording: int | None,
):
    classifier, scaler, saved_config = load_model_artifacts(model_dir)
    feature_names = list(saved_config.get("feature_names", FEATURE_NAMES))
    class_order = list(saved_config.get("class_order", FLIGHT_MODE_CLASSES))
    config = FeatureConfig.from_dict(saved_config)

    if segments_per_recording is not None:
        config = config.with_segments(segments_per_recording)

    feature_rows = extract_recording_feature_rows(
        file_path,
        config,
        require_metadata=False,
    )
    feature_matrix = feature_rows_to_matrix(feature_rows, feature_names)
    scaled_features = scale_features(feature_matrix, scaler)

    segment_predictions = [
        str(label) for label in classifier.predict(scaled_features)
    ]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(scaled_features)
        classifier_classes = [str(label) for label in classifier.classes_]
        mean_probabilities = {
            label: float(
                np.mean(probabilities[:, classifier_classes.index(label)])
            )
            for label in class_order
        }
    else:
        probabilities = None
        mean_probabilities = {
            label: float(segment_predictions.count(label) / len(segment_predictions))
            for label in class_order
        }

    baseline_prediction = majority_vote(segment_predictions, class_order)

    return (
        baseline_prediction,
        probabilities,
        feature_rows,
        segment_predictions,
        mean_probabilities,
        class_order,
    )


def _predict_temporal(
    file_path: Path,
    temporal_model_dir: Path,
):
    classifier, scaler, saved_config = load_model_artifacts(temporal_model_dir)

    feature_names = list(saved_config.get("feature_names", FEATURE_NAMES))
    config = FeatureConfig.from_dict(saved_config)
    config = config.with_segments(10)

    feature_rows = extract_recording_feature_rows(
        file_path,
        config,
        require_metadata=False,
    )
    feature_matrix = feature_rows_to_matrix(feature_rows, feature_names)
    scaled_features = scale_features(feature_matrix, scaler)
    probabilities = classifier.predict_proba(scaled_features)

    classifier_classes = [str(label) for label in classifier.classes_]
    ordered = np.zeros((len(probabilities), 3), dtype=float)
    for source_index, label in enumerate(classifier_classes):
        target_index = {"ON": 0, "HO": 1, "FY": 2}[label]
        ordered[:, target_index] = probabilities[:, source_index]

    recording_indices = {
        str(row.get("recording_index", ""))
        for row in feature_rows
    }
    recording_index = next(iter(recording_indices), "")

    if len(recording_indices) > 1:
        raise ValueError(
            f"Temporal prediction found multiple recording indices: {recording_indices}"
        )

    return ordered, recording_index


def predict_air_recording(
    file_path: Path,
    model_dir: Path,
    segments_per_recording: int | None = None,
    enable_optimization: bool = True,
    temporal_model_dir: Path | None = None,
) -> PredictionResult:
    """
    Preserve the original production prediction and optionally add the
    validated temporal optimization layer.
    """
    (
        baseline_prediction,
        _original_probabilities,
        feature_rows,
        segment_predictions,
        mean_probabilities,
        class_order,
    ) = _predict_original(
        file_path,
        model_dir,
        segments_per_recording,
    )

    optimized_prediction = baseline_prediction
    optimization_reason = "Optimization disabled"
    optimization_applied = False

    if enable_optimization:
        temporal_model_dir = temporal_model_dir or (model_dir / "temporal")

        required = [
            temporal_model_dir / "classifier.joblib",
            temporal_model_dir / "scaler.joblib",
            temporal_model_dir / "feature_config.json",
        ]

        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Temporal optimization model is not installed. Missing:\n"
                + "\n".join(missing)
            )

        probabilities, recording_index = _predict_temporal(
            file_path,
            temporal_model_dir,
        )

        optimized_prediction, optimization_reason = optimize_prediction(
            probabilities,
            baseline_prediction,
            recording_index,
        )
        optimization_applied = optimized_prediction != baseline_prediction

    final_prediction = optimized_prediction
    confidence = mean_probabilities.get(final_prediction, 0.0)

    return PredictionResult(
        file_path=file_path,
        final_prediction=final_prediction,
        final_prediction_label=FLIGHT_MODE_LABELS.get(
            final_prediction,
            final_prediction,
        ),
        confidence=confidence,
        vote_counts={
            label: segment_predictions.count(label)
            for label in class_order
        },
        mean_probabilities=mean_probabilities,
        feature_rows=feature_rows,
        segment_predictions=segment_predictions,
        baseline_prediction=baseline_prediction,
        optimized_prediction=optimized_prediction,
        optimization_applied=optimization_applied,
        optimization_reason=optimization_reason,
    )
