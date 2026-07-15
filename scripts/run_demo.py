"""
Run a folder-level demonstration of the AIR flight-mode predictor.

The script loads a saved model and analyses each AIR `.dat` recording in an
input folder:

    .dat -> IQ loading -> 20 ms segments -> RF features -> scaler
    -> Random Forest -> majority vote

The demo is intended for reproducible project presentation. It reports true
labels when they can be decoded from DroneDetect filenames/folders, but the
prediction pipeline itself can run without labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.iq_loader import FLIGHT_MODE_LABELS, parse_recording_metadata
from src.prediction import predict_air_recording


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASS_ORDER = ["ON", "HO", "FY"]


def discover_air_files(input_folder: Path) -> list[Path]:
    """
    Find AIR `.dat` files without requiring labels to be parseable.

    Args:
        input_folder: Folder to search recursively.

    Returns:
        Sorted AIR `.dat` paths.

    Raises:
        FileNotFoundError: If the folder does not exist or no AIR files exist.
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    files = sorted(input_folder.rglob("AIR*.dat"))
    if not files:
        raise FileNotFoundError(f"No AIR .dat files found under {input_folder}")
    return files


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """
    Write demo rows to CSV.

    Args:
        path: Destination CSV path.
        rows: Demo result rows.
        fieldnames: Ordered CSV columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, true_labels: list[str], predicted_labels: list[str]) -> None:
    """
    Save a confusion matrix image for labeled demo files.

    Args:
        path: Destination PNG path.
        true_labels: Decoded true flight modes.
        predicted_labels: Predicted flight modes.
    """
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_ORDER)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_ORDER)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("AIR Flight-Mode Demo Confusion Matrix")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_summary_report(
    path: Path,
    input_folder: Path,
    rows: list[dict[str, object]],
    accuracy: float | None,
    confusion_matrix_path: Path | None,
) -> None:
    """
    Create a concise Markdown summary for the demonstration run.

    Args:
        path: Destination Markdown path.
        input_folder: Demo input folder.
        rows: Demo result rows.
        accuracy: Optional accuracy over labeled recordings.
        confusion_matrix_path: Optional confusion matrix image path.
    """
    labeled_count = sum(1 for row in rows if row["true_label"])
    correct_count = sum(
        1
        for row in rows
        if row["true_label"] and row["true_label"] == row["predicted_label"]
    )
    class_lines = []
    for label in CLASS_ORDER:
        matching = [row for row in rows if row["predicted_label"] == label]
        class_lines.append(
            f"- `{label}` ({FLIGHT_MODE_LABELS[label]}): {len(matching)} predicted recording(s)"
        )

    accuracy_text = "Not available because no true labels were decoded."
    if accuracy is not None:
        accuracy_text = f"{accuracy * 100:.1f}% ({correct_count}/{labeled_count})"

    confusion_text = "Not generated because no labeled demo files were available."
    if confusion_matrix_path is not None:
        confusion_text = str(confusion_matrix_path)

    report = f"""# Demo Prediction Summary

## Purpose

This demonstration verifies that the saved engineering predictor can load AIR
RF `.dat` recordings, extract the same RF features used during training, apply
the saved scaler/classifier, and produce one flight-mode prediction per
recording by majority voting across 20 ms segments.

## Input

- Folder: `{input_folder}`
- Recordings analysed: {len(rows)}
- Labeled recordings: {labeled_count}

## Overall Demo Accuracy

{accuracy_text}

## Prediction Counts

{chr(10).join(class_lines)}

## Output Files

- Demo table: `outputs/demo_results.csv`
- Summary report: `outputs/demo_summary_report.md`
- Confusion matrix: `{confusion_text}`

## Engineering Interpretation

The demo uses passive RF IQ recordings only. The classifier is the final stage
of a larger sensing pipeline: IQ loading, DC removal, FFT/PSD/spectrogram
analysis, RF feature extraction, scaling, and then traditional machine learning.
Confidence is reported as the mean class probability across analysed segments.
"""
    path.write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line demo settings.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a folder-level AIR flight-mode prediction demo."
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing AIR .dat files.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--segments-per-recording",
        type=int,
        default=None,
        help=(
            "Override saved segment count. Use 100 for a full 2 s recording "
            "with non-overlapping 20 ms windows."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """
    Run prediction for all AIR files in a folder.

    Returns:
        None.

    Notes:
        This is a demonstration workflow, not the unbiased validation estimate.
        The unbiased metric remains the recording-grouped Stage 6 result.
    """
    args = parse_arguments()
    air_files = discover_air_files(args.input_folder)
    if args.max_files is not None:
        air_files = air_files[: args.max_files]

    rows: list[dict[str, object]] = []
    print("\nRF UAV Behaviour Predictor - Demo")
    print(f"Input folder: {args.input_folder}")
    print(f"AIR recordings found: {len(air_files)}")

    for index, file_path in enumerate(air_files, start=1):
        result = predict_air_recording(
            file_path=file_path,
            model_dir=args.model_dir,
            segments_per_recording=args.segments_per_recording,
        )
        try:
            metadata = parse_recording_metadata(file_path)
            true_label = metadata.mode_code
        except ValueError:
            true_label = ""

        row = {
            "file_name": file_path.name,
            "true_label": true_label,
            "predicted_label": result.final_prediction,
            "confidence": round(result.confidence * 100.0, 1),
            "segment_count": len(result.segment_predictions),
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(air_files):02d}] {file_path.name} "
            f"true={true_label or 'unknown'} predicted={result.final_prediction} "
            f"confidence={row['confidence']:.1f}%"
        )

    demo_csv = args.output_dir / "demo_results.csv"
    report_path = args.output_dir / "demo_summary_report.md"
    confusion_path = args.output_dir / "demo_confusion_matrix.png"
    write_csv(
        demo_csv,
        rows,
        ["file_name", "true_label", "predicted_label", "confidence", "segment_count"],
    )

    labeled_rows = [row for row in rows if row["true_label"]]
    accuracy = None
    matrix_path = None
    if labeled_rows:
        true_labels = [str(row["true_label"]) for row in labeled_rows]
        predicted_labels = [str(row["predicted_label"]) for row in labeled_rows]
        accuracy = float(accuracy_score(true_labels, predicted_labels))
        save_confusion_matrix(confusion_path, true_labels, predicted_labels)
        matrix_path = confusion_path

    write_summary_report(report_path, args.input_folder, rows, accuracy, matrix_path)

    print("\nDemo complete.")
    if accuracy is not None:
        print(f"Overall demo accuracy: {accuracy * 100:.1f}%")
    print(f"Saved results: {demo_csv.resolve()}")
    print(f"Saved report: {report_path.resolve()}")
    if matrix_path is not None:
        print(f"Saved confusion matrix: {matrix_path.resolve()}")


if __name__ == "__main__":
    main()
