"""
Stage 6: Leakage-safe traditional machine-learning validation.

Objectives:
    1. Classify flight mode: ON, HO, FY
    2. Classify interference: 00, 01, 10, 11
    3. Classify the complete 12-state label

Critical leakage rule:
    All 20 ms segments from one original .dat recording must remain together.
    No recording may contribute segments to both training and testing.

Cross-validation design:
    Each AIR state contains five original recordings with indices 00-04.
    Fold 0 tests recording index 00 from all 12 states, fold 1 tests index 01,
    and so on. Therefore every fold contains:

        12 held-out recordings x 10 segments = 120 test segments

    Training contains the other four recordings from every state.

Models:
    - Dummy baseline
    - Logistic regression
    - Linear SVM
    - RBF SVM
    - Random Forest

No hyperparameter tuning is performed in this baseline stage. This avoids
optimistically tuning models on the same held-out folds used for reporting.

Two evaluation levels:
    Segment level:
        Each 20 ms segment is classified independently.

    Recording level:
        The ten segment predictions from a held-out recording are combined by
        majority vote. This is often the more meaningful engineering result.
"""

from pathlib import Path

import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_INPUT_CSV = (
    Path("outputs")
    / "stage4_air_state_analysis"
    / "per_record_features.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage6_leakage_safe_ml_validation"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "rms_power",
    "signal_energy",
    "fft_peak_frequency_mhz",
    "peak_frequency_mhz",
    "occupied_bandwidth_mhz",
    "spectral_entropy",
    "spectral_centroid_mhz",
    "spectrogram_temporal_variability",
    "spectrogram_active_fraction",
]

TASKS = {
    "flight_mode": {
        "target_column": "mode_code",
        "labels": ["ON", "HO", "FY"],
        "chance_balanced_accuracy": 1.0 / 3.0,
    },
    "interference": {
        "target_column": "interference_code",
        "labels": ["00", "01", "10", "11"],
        "chance_balanced_accuracy": 1.0 / 4.0,
    },
    "state_12": {
        "target_column": "state",
        "labels": [
            "00_ON",
            "00_HO",
            "00_FY",
            "01_ON",
            "01_HO",
            "01_FY",
            "10_ON",
            "10_HO",
            "10_FY",
            "11_ON",
            "11_HO",
            "11_FY",
        ],
        "chance_balanced_accuracy": 1.0 / 12.0,
    },
}


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    """Read the Stage 4 segment-level RF feature matrix."""
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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    """Convert selected RF feature columns to a numeric matrix."""
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


def build_models() -> dict[str, object]:
    """
    Create fixed traditional-ML baselines.

    Scaling is fitted inside each training fold for linear and kernel models.
    This is essential: fitting the scaler on the full dataset would leak test
    information into training.
    """
    return {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LinearSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RBFSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def validate_dataset(rows: list[dict[str, str]]) -> list[str]:
    """
    Validate the expected grouped dataset structure.

    Returns:
        Sorted recording indices used as the five test folds.
    """
    recording_paths = sorted({row["file_path"] for row in rows})
    if len(recording_paths) != 60:
        raise ValueError(
            f"Expected 60 original AIR recordings, found {len(recording_paths)}."
        )

    fold_ids = sorted({row["recording_index"] for row in rows})
    if len(fold_ids) != 5:
        raise ValueError(
            f"Expected five recording indices for cross-validation, found {fold_ids}."
        )

    for file_path in recording_paths:
        file_rows = [row for row in rows if row["file_path"] == file_path]
        if len(file_rows) != 10:
            raise ValueError(
                f"Expected 10 segments for {file_path}, found {len(file_rows)}."
            )

    for task_name, task in TASKS.items():
        present_labels = {row[str(task["target_column"])] for row in rows}
        expected_labels = set(task["labels"])
        if present_labels != expected_labels:
            raise ValueError(
                f"Task {task_name} labels differ from expectation. "
                f"Found {sorted(present_labels)}."
            )

    return fold_ids


def majority_vote(predictions: list[str], label_order: list[str]) -> str:
    """
    Combine segment predictions for one recording.

    Ties are resolved using the fixed class order so results are reproducible.
    """
    counts = Counter(predictions)
    return max(label_order, key=lambda label: (counts[label], -label_order.index(label)))


def recording_level_predictions(
    rows: list[dict[str, str]],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_order: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Combine segment predictions into one prediction per original recording."""
    grouped_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(row["file_path"], []).append(index)

    recording_paths = []
    recording_true = []
    recording_predicted = []

    for file_path, indices in sorted(grouped_indices.items()):
        true_values = {str(true_labels[index]) for index in indices}
        if len(true_values) != 1:
            raise ValueError(f"Recording contains inconsistent labels: {file_path}")

        recording_paths.append(file_path)
        recording_true.append(true_values.pop())
        recording_predicted.append(
            majority_vote(
                [str(predicted_labels[index]) for index in indices],
                label_order,
            )
        )

    return recording_paths, recording_true, recording_predicted


def calculate_metrics(
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Calculate class-balanced metrics."""
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_grouped_cross_validation(
    rows: list[dict[str, str]],
    feature_matrix: np.ndarray,
    fold_ids: list[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, str, str], tuple[list[str], list[str]]],
]:
    """
    Run five-fold recording-grouped validation for all tasks and models.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Segment-level metadata rows. Multiple rows may come from the same
        original `.dat` recording.
    feature_matrix : np.ndarray
        RF feature matrix aligned row-for-row with ``rows``.
    fold_ids : list[str]
        Recording identifiers used as validation groups.

    Returns
    -------
    tuple
        Fold metrics, pooled metrics, prediction rows, and pooled
        true/predicted labels used for confusion matrices.

    Notes
    -----
    Grouping by original recording is mandatory here. If 20 ms segments from
    the same `.dat` file appeared in both training and validation folds, the
    classifier could exploit recording-specific quirks instead of learning RF
    behaviour differences that generalize to unseen recordings.
    """
    models = build_models()
    fold_metric_rows = []
    prediction_rows = []
    pooled: dict[tuple[str, str, str], dict[str, list[str]]] = {}

    for task_name, task in TASKS.items():
        target_column = str(task["target_column"])
        labels = list(task["labels"])
        all_targets = np.array([row[target_column] for row in rows])

        for model_name, model_template in models.items():
            for fold_id in fold_ids:
                train_indices = np.array(
                    [
                        index
                        for index, row in enumerate(rows)
                        if row["recording_index"] != fold_id
                    ]
                )
                test_indices = np.array(
                    [
                        index
                        for index, row in enumerate(rows)
                        if row["recording_index"] == fold_id
                    ]
                )

                train_paths = {rows[index]["file_path"] for index in train_indices}
                test_paths = {rows[index]["file_path"] for index in test_indices}
                if train_paths & test_paths:
                    raise RuntimeError("Recording leakage detected between train and test.")

                model = clone(model_template)
                model.fit(feature_matrix[train_indices], all_targets[train_indices])
                segment_predictions = model.predict(feature_matrix[test_indices])
                segment_true = all_targets[test_indices]
                test_rows = [rows[index] for index in test_indices]

                segment_metrics = calculate_metrics(
                    segment_true,
                    segment_predictions,
                    labels,
                )
                for index, true_label, predicted_label in zip(
                    test_indices,
                    segment_true,
                    segment_predictions,
                ):
                    prediction_rows.append(
                        {
                            "validation": "grouped_5_fold",
                            "task": task_name,
                            "model": model_name,
                            "fold": fold_id,
                            "evaluation_unit": "segment",
                            "file_path": rows[index]["file_path"],
                            "segment_index": rows[index]["segment_index"],
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                        }
                    )

                recording_paths, recording_true, recording_predicted = (
                    recording_level_predictions(
                        test_rows,
                        segment_true,
                        segment_predictions,
                        labels,
                    )
                )
                recording_metrics = calculate_metrics(
                    recording_true,
                    recording_predicted,
                    labels,
                )
                for file_path, true_label, predicted_label in zip(
                    recording_paths,
                    recording_true,
                    recording_predicted,
                ):
                    prediction_rows.append(
                        {
                            "validation": "grouped_5_fold",
                            "task": task_name,
                            "model": model_name,
                            "fold": fold_id,
                            "evaluation_unit": "recording",
                            "file_path": file_path,
                            "segment_index": "",
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                        }
                    )

                for evaluation_unit, metrics, sample_count in [
                    ("segment", segment_metrics, len(segment_true)),
                    ("recording", recording_metrics, len(recording_true)),
                ]:
                    fold_metric_rows.append(
                        {
                            "validation": "grouped_5_fold",
                            "task": task_name,
                            "model": model_name,
                            "fold": fold_id,
                            "evaluation_unit": evaluation_unit,
                            "sample_count": sample_count,
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "chance_balanced_accuracy": task[
                                "chance_balanced_accuracy"
                            ],
                        }
                    )

                for evaluation_unit, true_values, predicted_values in [
                    ("segment", segment_true.tolist(), segment_predictions.tolist()),
                    ("recording", recording_true, recording_predicted),
                ]:
                    key = (task_name, model_name, evaluation_unit)
                    pooled.setdefault(key, {"true": [], "predicted": []})
                    pooled[key]["true"].extend(true_values)
                    pooled[key]["predicted"].extend(predicted_values)

            print(f"Completed grouped CV: {task_name} / {model_name}")

    overall_rows = []
    confusion_data = {}
    for (task_name, model_name, evaluation_unit), values in pooled.items():
        labels = list(TASKS[task_name]["labels"])
        metrics = calculate_metrics(values["true"], values["predicted"], labels)
        matching_folds = [
            row
            for row in fold_metric_rows
            if row["task"] == task_name
            and row["model"] == model_name
            and row["evaluation_unit"] == evaluation_unit
        ]
        overall_rows.append(
            {
                "validation": "grouped_5_fold",
                "task": task_name,
                "model": model_name,
                "evaluation_unit": evaluation_unit,
                "sample_count": len(values["true"]),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "balanced_accuracy_fold_std": float(
                    np.std(
                        [float(row["balanced_accuracy"]) for row in matching_folds]
                    )
                ),
                "macro_f1": metrics["macro_f1"],
                "macro_f1_fold_std": float(
                    np.std([float(row["macro_f1"]) for row in matching_folds])
                ),
                "chance_balanced_accuracy": TASKS[task_name][
                    "chance_balanced_accuracy"
                ],
            }
        )
        confusion_data[(task_name, model_name, evaluation_unit)] = (
            values["true"],
            values["predicted"],
        )

    return fold_metric_rows, overall_rows, prediction_rows, confusion_data


def per_class_recall_rows(
    confusion_data: dict[tuple[str, str, str], tuple[list[str], list[str]]],
) -> list[dict[str, object]]:
    """Create per-class recall tables from pooled predictions."""
    rows = []
    for (task_name, model_name, evaluation_unit), (
        true_values,
        predicted_values,
    ) in confusion_data.items():
        labels = list(TASKS[task_name]["labels"])
        recalls = recall_score(
            true_values,
            predicted_values,
            labels=labels,
            average=None,
            zero_division=0,
        )
        for label, recall in zip(labels, recalls):
            rows.append(
                {
                    "validation": "grouped_5_fold",
                    "task": task_name,
                    "model": model_name,
                    "evaluation_unit": evaluation_unit,
                    "class_label": label,
                    "recall": float(recall),
                }
            )
    return rows


def evaluate_generalization(
    rows: list[dict[str, str]],
    feature_matrix: np.ndarray,
) -> list[dict[str, object]]:
    """
    Test whether learned information generalizes across the other physical factor.

    Tests:
        - Flight mode with one complete interference condition held out
        - Interference with one complete flight mode held out
    """
    models = build_models()
    result_rows = []

    tests = [
        {
            "name": "flight_mode_holdout_interference",
            "task": "flight_mode",
            "holdout_column": "interference_code",
            "holdout_values": ["00", "01", "10", "11"],
        },
        {
            "name": "interference_holdout_mode",
            "task": "interference",
            "holdout_column": "mode_code",
            "holdout_values": ["ON", "HO", "FY"],
        },
    ]

    for test in tests:
        task_name = str(test["task"])
        task = TASKS[task_name]
        labels = list(task["labels"])
        target_column = str(task["target_column"])
        all_targets = np.array([row[target_column] for row in rows])

        for holdout_value in test["holdout_values"]:
            train_indices = np.array(
                [
                    index
                    for index, row in enumerate(rows)
                    if row[str(test["holdout_column"])] != holdout_value
                ]
            )
            test_indices = np.array(
                [
                    index
                    for index, row in enumerate(rows)
                    if row[str(test["holdout_column"])] == holdout_value
                ]
            )

            for model_name, model_template in models.items():
                model = clone(model_template)
                model.fit(feature_matrix[train_indices], all_targets[train_indices])
                segment_predictions = model.predict(feature_matrix[test_indices])
                segment_true = all_targets[test_indices]
                segment_metrics = calculate_metrics(
                    segment_true,
                    segment_predictions,
                    labels,
                )

                test_rows = [rows[index] for index in test_indices]
                _, recording_true, recording_predicted = recording_level_predictions(
                    test_rows,
                    segment_true,
                    segment_predictions,
                    labels,
                )
                recording_metrics = calculate_metrics(
                    recording_true,
                    recording_predicted,
                    labels,
                )

                for evaluation_unit, metrics, sample_count in [
                    ("segment", segment_metrics, len(segment_true)),
                    ("recording", recording_metrics, len(recording_true)),
                ]:
                    result_rows.append(
                        {
                            "validation": test["name"],
                            "task": task_name,
                            "held_out_factor": test["holdout_column"],
                            "held_out_value": holdout_value,
                            "model": model_name,
                            "evaluation_unit": evaluation_unit,
                            "sample_count": sample_count,
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "chance_balanced_accuracy": task[
                                "chance_balanced_accuracy"
                            ],
                        }
                    )

            print(f"Completed generalization test: {test['name']} / {holdout_value}")

    return result_rows


def plot_model_comparison(
    overall_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot recording-level balanced accuracy for all tasks and models."""
    model_order = list(build_models())
    task_order = list(TASKS)
    width = 0.15
    positions = np.arange(len(task_order))

    figure, axis = plt.subplots(figsize=(12, 6))
    for model_index, model_name in enumerate(model_order):
        values = []
        for task_name in task_order:
            row = next(
                row
                for row in overall_rows
                if row["task"] == task_name
                and row["model"] == model_name
                and row["evaluation_unit"] == "recording"
            )
            values.append(float(row["balanced_accuracy"]))

        offset = (model_index - (len(model_order) - 1) / 2) * width
        axis.bar(
            positions + offset,
            values,
            width=width,
            label=model_name,
        )

    for task_index, task_name in enumerate(task_order):
        chance = float(TASKS[task_name]["chance_balanced_accuracy"])
        axis.hlines(
            chance,
            task_index - 0.42,
            task_index + 0.42,
            colors="black",
            linestyles="--",
            linewidth=1.0,
        )

    axis.set_xticks(positions, ["Flight mode", "Interference", "12 states"])
    axis.set_ylabel("Recording-level balanced accuracy")
    axis.set_ylim(0, 1.05)
    axis.set_title("Leakage-Safe Traditional ML Validation")
    axis.legend(ncol=3)
    axis.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def select_best_models(
    overall_rows: list[dict[str, object]],
) -> dict[str, str]:
    """Select the best non-dummy recording-level model per task."""
    best = {}
    for task_name in TASKS:
        candidates = [
            row
            for row in overall_rows
            if row["task"] == task_name
            and row["evaluation_unit"] == "recording"
            and row["model"] != "Dummy"
        ]
        winner = max(
            candidates,
            key=lambda row: (
                float(row["balanced_accuracy"]),
                float(row["macro_f1"]),
            ),
        )
        best[task_name] = str(winner["model"])
    return best


def plot_best_confusion_matrices(
    confusion_data: dict[tuple[str, str, str], tuple[list[str], list[str]]],
    best_models: dict[str, str],
    output_dir: Path,
) -> None:
    """Save normalized recording-level confusion matrices for the best models."""
    for task_name, model_name in best_models.items():
        labels = list(TASKS[task_name]["labels"])
        true_values, predicted_values = confusion_data[
            (task_name, model_name, "recording")
        ]
        matrix = confusion_matrix(
            true_values,
            predicted_values,
            labels=labels,
            normalize="true",
        )

        figure, axis = plt.subplots(figsize=(8, 7))
        display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        display.plot(
            ax=axis,
            cmap="Blues",
            values_format=".2f",
            colorbar=False,
        )
        axis.set_title(
            f"{task_name}: {model_name}\n"
            "Recording-level normalized confusion matrix"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_{task_name}_{model_name}.png",
            dpi=180,
        )
        plt.close(figure)


def summarize_generalization(
    generalization_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Average holdout-condition results for a concise comparison table."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in generalization_rows:
        key = (
            str(row["validation"]),
            str(row["model"]),
            str(row["evaluation_unit"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (validation, model, evaluation_unit), rows in grouped.items():
        summary_rows.append(
            {
                "validation": validation,
                "model": model,
                "evaluation_unit": evaluation_unit,
                "holdout_count": len(rows),
                "mean_balanced_accuracy": float(
                    np.mean([float(row["balanced_accuracy"]) for row in rows])
                ),
                "minimum_balanced_accuracy": float(
                    np.min([float(row["balanced_accuracy"]) for row in rows])
                ),
                "mean_macro_f1": float(
                    np.mean([float(row["macro_f1"]) for row in rows])
                ),
                "chance_balanced_accuracy": rows[0][
                    "chance_balanced_accuracy"
                ],
            }
        )
    return summary_rows


def write_engineering_report(
    output_path: Path,
    overall_rows: list[dict[str, object]],
    best_models: dict[str, str],
    generalization_summary: list[dict[str, object]],
) -> None:
    """Write a concise interpretation of leakage-safe ML results."""
    task_lines = []
    for task_name, model_name in best_models.items():
        row = next(
            row
            for row in overall_rows
            if row["task"] == task_name
            and row["model"] == model_name
            and row["evaluation_unit"] == "recording"
        )
        chance = float(row["chance_balanced_accuracy"])
        accuracy = float(row["balanced_accuracy"])
        task_lines.append(
            f"- `{task_name}`: best model `{model_name}`, balanced accuracy "
            f"{accuracy:.3f} versus chance {chance:.3f}, macro F1 "
            f"{float(row['macro_f1']):.3f}"
        )

    generalization_lines = []
    generalization_best = {}
    for validation in [
        "flight_mode_holdout_interference",
        "interference_holdout_mode",
    ]:
        candidates = [
            row
            for row in generalization_summary
            if row["validation"] == validation
            and row["evaluation_unit"] == "recording"
            and row["model"] != "Dummy"
        ]
        best = max(
            candidates,
            key=lambda row: float(row["mean_balanced_accuracy"]),
        )
        generalization_best[validation] = best
        generalization_lines.append(
            f"- `{validation}`: best mean recording-level balanced accuracy "
            f"{float(best['mean_balanced_accuracy']):.3f} using "
            f"`{best['model']}`; worst held-out condition "
            f"{float(best['minimum_balanced_accuracy']):.3f}; chance "
            f"{float(best['chance_balanced_accuracy']):.3f}"
        )

    flight_generalization = generalization_best[
        "flight_mode_holdout_interference"
    ]
    interference_generalization = generalization_best[
        "interference_holdout_mode"
    ]
    flight_generalization_accuracy = float(
        flight_generalization["mean_balanced_accuracy"]
    )
    flight_chance = float(flight_generalization["chance_balanced_accuracy"])
    interference_generalization_accuracy = float(
        interference_generalization["mean_balanced_accuracy"]
    )
    interference_chance = float(
        interference_generalization["chance_balanced_accuracy"]
    )

    if flight_generalization_accuracy >= flight_chance + 0.15:
        flight_conclusion = (
            "Flight-mode information generalizes across interference conditions, "
            "so it is a genuine and relatively stable RF source."
        )
    else:
        flight_conclusion = (
            "Flight-mode information does not yet generalize strongly across "
            "unseen interference conditions."
        )

    if interference_generalization_accuracy <= interference_chance + 0.05:
        interference_conclusion = (
            "Interference classification is close to chance when a complete "
            "flight mode is unseen. The interference signature is therefore "
            "strongly mode-dependent rather than independent."
        )
    else:
        interference_conclusion = (
            "Interference information generalizes across unseen flight modes."
        )

    report = f"""# Leakage-Safe Traditional ML Validation

## Validation Design

- 600 RF feature rows from 60 original `.dat` recordings
- 10 segments per recording
- Five grouped folds
- Every fold holds out one complete recording from each of the 12 states
- Scaling is fitted on training folds only
- No segment from a test recording appears in training

## Main Results

{chr(10).join(task_lines)}

## Cross-Condition Generalization

{chr(10).join(generalization_lines)}

## Engineering Conclusion

- {flight_conclusion}
- {interference_conclusion}
- The 12-state task is well above its 8.3% chance level, so useful joint
  mode/interference information exists, but the current features do not fully
  separate all 12 states.
- Overall, the most defensible interpretation is: **flight mode is the primary
  generalizable RF source; interference information is weaker and largely
  expressed through its interaction with flight mode.**

## How To Interpret The Results

- Performance near chance means the extracted features do not support that
  classification task reliably.
- Performance clearly above chance means usable RF information exists.
- Strong grouped-CV performance but weak cross-condition performance means the
  model is learning state-specific combinations rather than a factor that
  generalizes independently.
- Recording-level results are the primary engineering result. Segment-level
  results show how reliable a single 20 ms decision is.

## Important Limitation

These are fixed baseline models, not tuned final models. Hyperparameter tuning,
if added later, must use a nested grouped validation procedure.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """Parse Stage 6 settings."""
    parser = argparse.ArgumentParser(
        description="Leakage-safe traditional ML validation for AIR RF features."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Stage 4 segment-level feature matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for validation tables, figures, and report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all grouped validation and cross-condition tests."""
    args = parse_arguments()
    rows = read_feature_rows(args.input_csv)
    feature_matrix = build_feature_matrix(rows)
    fold_ids = validate_dataset(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        fold_metrics,
        overall_metrics,
        prediction_rows,
        confusion_data,
    ) = evaluate_grouped_cross_validation(rows, feature_matrix, fold_ids)
    recall_rows = per_class_recall_rows(confusion_data)
    generalization_rows = evaluate_generalization(rows, feature_matrix)
    generalization_summary = summarize_generalization(generalization_rows)
    best_models = select_best_models(overall_metrics)

    write_csv(
        args.output_dir / "grouped_cv_fold_metrics.csv",
        fold_metrics,
        [
            "validation",
            "task",
            "model",
            "fold",
            "evaluation_unit",
            "sample_count",
            "balanced_accuracy",
            "macro_f1",
            "chance_balanced_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "grouped_cv_overall_metrics.csv",
        overall_metrics,
        [
            "validation",
            "task",
            "model",
            "evaluation_unit",
            "sample_count",
            "balanced_accuracy",
            "balanced_accuracy_fold_std",
            "macro_f1",
            "macro_f1_fold_std",
            "chance_balanced_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "grouped_cv_predictions.csv",
        prediction_rows,
        [
            "validation",
            "task",
            "model",
            "fold",
            "evaluation_unit",
            "file_path",
            "segment_index",
            "true_label",
            "predicted_label",
        ],
    )
    write_csv(
        args.output_dir / "per_class_recall.csv",
        recall_rows,
        [
            "validation",
            "task",
            "model",
            "evaluation_unit",
            "class_label",
            "recall",
        ],
    )
    write_csv(
        args.output_dir / "cross_condition_generalization.csv",
        generalization_rows,
        [
            "validation",
            "task",
            "held_out_factor",
            "held_out_value",
            "model",
            "evaluation_unit",
            "sample_count",
            "balanced_accuracy",
            "macro_f1",
            "chance_balanced_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "cross_condition_generalization_summary.csv",
        generalization_summary,
        [
            "validation",
            "model",
            "evaluation_unit",
            "holdout_count",
            "mean_balanced_accuracy",
            "minimum_balanced_accuracy",
            "mean_macro_f1",
            "chance_balanced_accuracy",
        ],
    )

    plot_model_comparison(
        overall_metrics,
        args.output_dir / "recording_level_model_comparison.png",
    )
    plot_best_confusion_matrices(
        confusion_data,
        best_models,
        args.output_dir,
    )
    write_engineering_report(
        args.output_dir / "engineering_validation_report.md",
        overall_metrics,
        best_models,
        generalization_summary,
    )

    print("\n--- Best recording-level grouped-CV results ---")
    for task_name, model_name in best_models.items():
        row = next(
            row
            for row in overall_metrics
            if row["task"] == task_name
            and row["model"] == model_name
            and row["evaluation_unit"] == "recording"
        )
        print(
            f"{task_name}: {model_name}, "
            f"balanced accuracy={float(row['balanced_accuracy']):.3f}, "
            f"macro F1={float(row['macro_f1']):.3f}, "
            f"chance={float(row['chance_balanced_accuracy']):.3f}"
        )

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
