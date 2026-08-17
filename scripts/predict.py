"""
Predict AIR UAV flight mode from one DroneDetect `.dat` recording.

This is the deployment command-line entry point:

    .dat -> IQ segments -> RF features -> saved scaler -> saved classifier
    -> segment votes -> final ON/HO/FY behaviour prediction

The true label is only decoded for demonstration when it exists in the path.
It is not used as a model input.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import FEATURE_NAMES
from src.iq_loader import FLIGHT_MODE_LABELS, parse_recording_metadata
from src.prediction import predict_air_recording


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "engineering_predictions"


def write_prediction_csv(output_path: Path, result) -> None:
    """
    Save segment-level features and predictions for engineering inspection.

    Args:
        output_path: Destination CSV path.
        result: `PredictionResult` returned by `predict_air_recording`.
    """
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
        for row, prediction in zip(result.feature_rows, result.segment_predictions):
            output_row = {
                "file_path": row["file_path"],
                "segment_index": row["segment_index"],
                "segment_start_sample": row["segment_start_sample"],
                "segment_start_ms": row["segment_start_ms"],
                "segment_duration_ms": row["segment_duration_ms"],
                "segment_prediction": prediction,
            }
            for feature_name in FEATURE_NAMES:
                output_row[feature_name] = row[feature_name]
            writer.writerow(output_row)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line prediction arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Predict AIR UAV behaviour state ON/HO/FY from RF IQ .dat data."
    )
    parser.add_argument("--file", type=Path, required=True, help="AIR .dat recording.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--segments-per-recording",
        type=int,
        default=None,
        help=(
            "Override saved segment count. Use 100 for a full 2 s recording "
            "with non-overlapping 20 ms windows, or 0 for all available windows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run prediction for one `.dat` file and print a human-readable summary.

    Returns:
        None.
    """
    args = parse_arguments()
    if not args.file.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.file}")

    result = predict_air_recording(
        file_path=args.file,
        model_dir=args.model_dir,
        segments_per_recording=args.segments_per_recording,
    )

    output_csv = args.output_dir / f"{args.file.stem}_flight_mode_predictions.csv"
    write_prediction_csv(output_csv, result)

    print("\n--------------------------------")
    print("RF UAV Behaviour Predictor")
    print("\nInput:")
    print(args.file.name)
    print("\nSegments analysed:")
    print(len(result.segment_predictions))
    print("\nPrediction:\n")
    print("Flight Mode:")
    print("\nOptimization:")
    print(f"Baseline prediction: {result.baseline_prediction}")
    print(f"Optimized prediction: {result.optimized_prediction}")
    print(f"Optimization applied: {result.optimization_applied}")
    print(f"Optimization reason: {result.optimization_reason}")
    print(result.final_prediction_label)
    print("\nConfidence:")
    print(f"{result.confidence * 100:.1f}%")
    print("\nVotes:\n")
    for label in ["ON", "HO", "FY"]:
        print(f"{label}:")
        print(result.vote_counts[label])
        print()
    print("Mean class probabilities:")
    for label in ["ON", "HO", "FY"]:
        print(f"  {label} ({FLIGHT_MODE_LABELS[label]}): {result.mean_probabilities[label]:.3f}")

    try:
        metadata = parse_recording_metadata(args.file)
    except ValueError:
        metadata = None
    if metadata is not None:
        print("\nLabel decoded from filename/path:")
        print(f"{metadata.mode_code} ({metadata.mode_label})")
        print(f"Prediction matches label: {result.final_prediction == metadata.mode_code}")

    print("\nSaved segment prediction table:")
    print(output_csv.resolve())
    print("--------------------------------")


if __name__ == "__main__":
    main()
