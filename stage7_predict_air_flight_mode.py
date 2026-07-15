"""
Stage 7B: Predict AIR flight mode for one .dat recording.

Prediction target:
    ON / HO / FY

Input:
    A DroneDetect-style AIR .dat file containing interleaved float32 IQ samples.

Processing:
    1. Select 10 evenly spaced 20 ms segments across the recording.
    2. Extract the same RF features used during training.
    3. Predict the flight mode for every segment.
    4. Combine segment predictions using majority vote.
    5. Report the final recording-level prediction and class probabilities.

This script does not use deep learning. It uses the saved traditional ML model
trained by stage7_train_flight_mode_predictor.py.
"""

from pathlib import Path

import argparse
import csv

import joblib
import numpy as np

from stage1_load_air_iq import count_raw_floats_from_file_size
from stage4_air_state_analysis import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SEGMENT_MS,
    DEFAULT_SEGMENTS_PER_FILE,
    DEFAULT_PSD_NPERSEG,
    DEFAULT_PSD_OVERLAP,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SPECTROGRAM_NPERSEG,
    DEFAULT_SPECTROGRAM_OVERLAP,
    DEFAULT_DC_EXCLUSION_HZ,
    FEATURE_NAMES,
    choose_evenly_spaced_segment_starts,
    extract_features,
    find_air_dat_files,
    load_iq_segment,
    parse_recording_metadata,
)


DEFAULT_MODEL_PATH = (
    Path("outputs")
    / "stage7_prediction_model"
    / "flight_mode_predictor.joblib"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage7_predictions"


def choose_input_file(file_path: str | None, dataset_root: Path) -> Path:
    """
    Select a file to predict.

    If no file is supplied, use the first AIR file found. This is convenient
    for testing the prediction pipeline, but real use should pass --file.
    """
    if file_path is not None:
        selected = Path(file_path)
        if not selected.exists():
            raise FileNotFoundError(f"Input .dat file does not exist: {selected}")
        return selected

    air_files = find_air_dat_files(dataset_root)
    if not air_files:
        raise FileNotFoundError(f"No AIR .dat files found under {dataset_root}")

    print("No --file supplied; using the first AIR file for demonstration:")
    print(f"  {air_files[0]}")
    return air_files[0]


def load_model_bundle(model_path: Path) -> dict[str, object]:
    """Load the trained model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run stage7_train_flight_mode_predictor.py first."
        )
    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_names", "class_labels"}
    missing = required_keys - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")
    return bundle


def extract_recording_features(
    file_path: Path,
    segment_ms: float,
    segments_per_file: int,
    sample_rate_hz: float,
    psd_nperseg: int,
    psd_overlap: int,
    spectrogram_nperseg: int,
    spectrogram_overlap: int,
    dc_exclusion_hz: float,
) -> list[dict[str, object]]:
    """
    Extract one RF feature row per selected segment from a .dat recording.

    RF meaning:
        Prediction is made from repeated short observations of the signal.
        The majority vote is more reliable than trusting one arbitrary 20 ms
        segment.
    """
    total_raw_floats = count_raw_floats_from_file_size(file_path)
    total_iq_samples = total_raw_floats // 2
    segment_samples = int(round(segment_ms / 1000.0 * sample_rate_hz))
    segment_starts = choose_evenly_spaced_segment_starts(
        total_iq_samples,
        segment_samples,
        first_start_sample=0,
        segments_per_file=segments_per_file,
    )

    rows = []
    for segment_index, segment_start in enumerate(segment_starts):
        samples_to_read = min(segment_samples, total_iq_samples - segment_start)
        iq_signal = load_iq_segment(file_path, segment_start, samples_to_read)
        features = extract_features(
            iq_signal,
            sample_rate_hz,
            psd_nperseg,
            psd_overlap,
            spectrogram_nperseg,
            spectrogram_overlap,
            dc_exclusion_hz,
        )
        row: dict[str, object] = {
            "file_path": str(file_path),
            "segment_index": segment_index,
            "segment_start_sample": segment_start,
            "segment_start_ms": segment_start / sample_rate_hz * 1e3,
            "segment_duration_ms": segment_ms,
        }
        row.update(features)
        rows.append(row)

    return rows


def build_feature_matrix(
    feature_rows: list[dict[str, object]],
    feature_names: list[str],
) -> np.ndarray:
    """Build model input matrix using the saved feature order."""
    matrix = np.array(
        [
            [float(row[feature_name]) for feature_name in feature_names]
            for row in feature_rows
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Prediction feature matrix contains NaN or infinite values.")
    return matrix


def majority_vote(labels: list[str], class_order: list[str]) -> str:
    """Return the most common class, with deterministic tie-breaking."""
    counts = {label: labels.count(label) for label in class_order}
    return max(class_order, key=lambda label: (counts[label], -class_order.index(label)))


def predict_recording(
    bundle: dict[str, object],
    feature_rows: list[dict[str, object]],
) -> tuple[str, list[dict[str, object]], dict[str, float]]:
    """Predict segment-level and recording-level flight mode."""
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    class_labels = list(bundle["class_labels"])
    feature_matrix = build_feature_matrix(feature_rows, feature_names)

    segment_predictions = [str(label) for label in model.predict(feature_matrix)]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_matrix)
        class_order = [str(label) for label in model.classes_]
        mean_probabilities = {
            class_label: float(np.mean(probabilities[:, class_order.index(class_label)]))
            for class_label in class_labels
        }
    else:
        mean_probabilities = {
            class_label: float(segment_predictions.count(class_label) / len(segment_predictions))
            for class_label in class_labels
        }

    final_prediction = majority_vote(segment_predictions, class_labels)

    prediction_rows = []
    for row, prediction in zip(feature_rows, segment_predictions):
        output_row = dict(row)
        output_row["segment_prediction"] = prediction
        prediction_rows.append(output_row)

    return final_prediction, prediction_rows, mean_probabilities


def write_prediction_csv(
    output_path: Path,
    prediction_rows: list[dict[str, object]],
) -> None:
    """Save segment-level prediction details for inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "segment_index",
        "segment_start_sample",
        "segment_start_ms",
        "segment_duration_ms",
        *FEATURE_NAMES,
        "segment_prediction",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)


def parse_arguments() -> argparse.Namespace:
    """Parse prediction settings."""
    parser = argparse.ArgumentParser(
        description="Predict AIR flight mode from a DroneDetect .dat recording."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the AIR .dat recording to classify.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used only when --file is omitted.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained model bundle path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where segment predictions are saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Predict one AIR recording."""
    args = parse_arguments()
    file_path = choose_input_file(args.file, args.data_root)
    bundle = load_model_bundle(args.model_path)

    segment_ms = float(bundle.get("segment_ms", DEFAULT_SEGMENT_MS))
    segments_per_file = int(bundle.get("segments_per_file", DEFAULT_SEGMENTS_PER_FILE))
    sample_rate_hz = float(bundle.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ))
    psd_nperseg = int(bundle.get("psd_nperseg", DEFAULT_PSD_NPERSEG))
    psd_overlap = int(bundle.get("psd_overlap", DEFAULT_PSD_OVERLAP))
    spectrogram_nperseg = int(
        bundle.get("spectrogram_nperseg", DEFAULT_SPECTROGRAM_NPERSEG)
    )
    spectrogram_overlap = int(
        bundle.get("spectrogram_overlap", DEFAULT_SPECTROGRAM_OVERLAP)
    )
    dc_exclusion_hz = float(bundle.get("dc_exclusion_hz", DEFAULT_DC_EXCLUSION_HZ))

    feature_rows = extract_recording_features(
        file_path,
        segment_ms,
        segments_per_file,
        sample_rate_hz,
        psd_nperseg,
        psd_overlap,
        spectrogram_nperseg,
        spectrogram_overlap,
        dc_exclusion_hz,
    )
    final_prediction, prediction_rows, mean_probabilities = predict_recording(
        bundle,
        feature_rows,
    )

    metadata = parse_recording_metadata(file_path)
    true_mode = metadata.get("mode_code")
    output_csv = args.output_dir / f"{file_path.stem}_flight_mode_predictions.csv"
    write_prediction_csv(output_csv, prediction_rows)

    print("\n--- AIR flight-mode prediction ---")
    print(f"Input file: {file_path}")
    print(f"Segments analysed: {len(prediction_rows)}")
    print(f"Final predicted flight mode: {final_prediction}")
    if true_mode is not None:
        print(f"Mode decoded from filename/path: {true_mode}")
        print(f"Prediction matches filename/path: {final_prediction == true_mode}")
    print("\nMean class probabilities across segments:")
    for label, probability in sorted(mean_probabilities.items()):
        print(f"  {label}: {probability:.3f}")
    print(f"\nSaved segment predictions: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
