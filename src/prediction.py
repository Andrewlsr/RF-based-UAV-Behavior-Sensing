"""
Deployment prediction pipeline for new AIR `.dat` recordings.

This module loads saved model artifacts, extracts the identical RF feature set
used during training, applies the saved scaler, predicts each 20 ms segment,
and combines segment predictions into one flight-mode decision.

Notes:
    Prediction intentionally does not require true labels in the filename.
    Labels may be decoded for demo reporting, but the classifier input is only
    the numeric RF feature matrix.
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


@dataclass(frozen=True)
class PredictionResult:
    """
    Recording-level prediction plus segment-level details.

    Attributes:
        file_path: Input `.dat` path.
        final_prediction: Final class code, one of `ON`, `HO`, or `FY`.
        final_prediction_label: Human-readable label.
        confidence: Mean model probability for the final predicted class.
        vote_counts: Segment vote count per class.
        mean_probabilities: Mean class probabilities across analysed segments.
        feature_rows: Segment-level RF features and metadata.
        segment_predictions: Per-segment predicted labels.
    """

    file_path: Path
    final_prediction: str
    final_prediction_label: str
    confidence: float
    vote_counts: dict[str, int]
    mean_probabilities: dict[str, float]
    feature_rows: list[dict[str, object]]
    segment_predictions: list[str]


def predict_air_recording(
    file_path: Path,
    model_dir: Path,
    segments_per_recording: int | None = None,
) -> PredictionResult:
    """
    Predict ON/HO/FY flight mode for one AIR .dat file.

    Args:
        file_path: Path to an AIR-style interleaved float32 IQ recording.
        model_dir: Folder containing saved model artifacts.
        segments_per_recording: Optional override for the saved segment count.

    Returns:
        `PredictionResult` containing the final majority-vote decision and
        segment-level details.

    Notes:
        Pipeline:
        .dat file -> IQ segments -> RF features -> scaler -> Random Forest
        -> segment predictions -> majority-vote recording prediction.

        Multiple 20 ms segments are analysed because one segment can be an
        unusual quiet/bursty moment. Majority voting makes the final recording
        decision less sensitive to one atypical slice. Confidence is aggregated
        as the mean predicted probability over all analysed segments.
    """
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
    segment_predictions = [str(label) for label in classifier.predict(scaled_features)]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(scaled_features)
        classifier_classes = [str(label) for label in classifier.classes_]
        mean_probabilities = {
            label: float(np.mean(probabilities[:, classifier_classes.index(label)]))
            for label in class_order
        }
    else:
        mean_probabilities = {
            label: float(segment_predictions.count(label) / len(segment_predictions))
            for label in class_order
        }

    vote_counts = {label: segment_predictions.count(label) for label in class_order}
    final_prediction = majority_vote(segment_predictions, class_order)
    confidence = mean_probabilities.get(final_prediction, 0.0)

    return PredictionResult(
        file_path=file_path,
        final_prediction=final_prediction,
        final_prediction_label=FLIGHT_MODE_LABELS.get(final_prediction, final_prediction),
        confidence=confidence,
        vote_counts=vote_counts,
        mean_probabilities=mean_probabilities,
        feature_rows=feature_rows,
        segment_predictions=segment_predictions,
    )
