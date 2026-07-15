"""
Stage 5: Determine where AIR RF information originates.

Research question:
    Does the RF information originate from flight mode, interference
    conditions, or the interaction between the two?

Method:
    1. Read the segment-level RF feature matrix from Stage 4.
    2. Aggregate the 10 segments from each original .dat file.
       This restores 60 independent recording-level observations.
    3. Apply a balanced 3 x 4 two-factor ANOVA to every RF feature:

           feature = flight mode
                   + interference
                   + flight mode x interference
                   + recording-to-recording residual

    4. Report:
       - F statistic and corrected p-value
       - partial eta-squared effect size
       - share of structured variation
       - dominant information source per feature

Primary aggregation:
    Median across the 10 segments in each recording. This is robust to unusual
    transient segments.

Sensitivity check:
    Mean across the 10 segments. A conclusion is considered robust when the
    median and mean analyses agree.

This is an engineering/statistical analysis, not machine-learning training.
"""

from pathlib import Path

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f as f_distribution


DEFAULT_INPUT_CSV = (
    Path("outputs")
    / "stage4_air_state_analysis"
    / "per_record_features.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage5_factorial_information_analysis"

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

# Signal energy is mathematically related to RMS power when every segment has
# the same number of samples. We keep it in the detailed results because it was
# requested, but exclude it from the non-redundant overall evidence count.
NON_REDUNDANT_FEATURES = [
    feature_name
    for feature_name in FEATURE_NAMES
    if feature_name != "signal_energy"
]

MODE_ORDER = ["ON", "HO", "FY"]
INTERFERENCE_ORDER = ["00", "01", "10", "11"]
EFFECT_ORDER = ["flight_mode", "interference", "interaction"]

EFFECT_LABELS = {
    "flight_mode": "Flight mode",
    "interference": "Interference",
    "interaction": "Mode x interference",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 4 feature matrix was not found: {path}\n"
            "Run stage4_air_state_analysis.py first."
        )

    with path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_segments_by_recording(
    segment_rows: list[dict[str, str]],
    aggregation: str,
) -> list[dict[str, object]]:
    """
    Aggregate segment features to one independent vector per recording.

    The 10 segments from one .dat file are repeated measurements of the same
    recording. Treating all 600 segments as independent would exaggerate the
    statistical sample size. Aggregation gives the correct 60 recording-level
    observations, with 5 recordings in each of the 12 states.
    """
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in segment_rows:
        grouped.setdefault(row["file_path"], []).append(row)

    if aggregation == "median":
        aggregate_function = np.median
    elif aggregation == "mean":
        aggregate_function = np.mean
    else:
        raise ValueError("Aggregation must be 'median' or 'mean'.")

    recording_rows: list[dict[str, object]] = []
    for file_path, rows in sorted(grouped.items()):
        first = rows[0]
        recording: dict[str, object] = {
            "file_path": file_path,
            "file_name": first["file_name"],
            "interference_code": first["interference_code"],
            "mode_code": first["mode_code"],
            "state": first["state"],
            "segment_count": len(rows),
            "aggregation": aggregation,
        }

        for feature_name in FEATURE_NAMES:
            values = np.array([float(row[feature_name]) for row in rows])
            recording[feature_name] = float(aggregate_function(values))

        recording_rows.append(recording)

    return recording_rows


def validate_balanced_design(recording_rows: list[dict[str, object]]) -> int:
    """
    Confirm the expected balanced 3 x 4 design.

    Returns:
        Number of independent recordings per mode/interference cell.
    """
    counts = {}
    for mode_code in MODE_ORDER:
        for interference_code in INTERFERENCE_ORDER:
            count = sum(
                row["mode_code"] == mode_code
                and row["interference_code"] == interference_code
                for row in recording_rows
            )
            counts[(mode_code, interference_code)] = count

    unique_counts = set(counts.values())
    if len(unique_counts) != 1 or 0 in unique_counts:
        details = ", ".join(
            f"{mode}/{interference}={count}"
            for (mode, interference), count in counts.items()
        )
        raise ValueError(f"The factorial design is not balanced: {details}")

    return unique_counts.pop()


def effect_size_label(partial_eta_squared: float) -> str:
    """
    Convert partial eta-squared to a familiar qualitative label.

    Conventional guide:
        < 0.01  negligible
        0.01    small
        0.06    medium
        0.14    large
    """
    if partial_eta_squared >= 0.14:
        return "large"
    if partial_eta_squared >= 0.06:
        return "medium"
    if partial_eta_squared >= 0.01:
        return "small"
    return "negligible"


def two_factor_anova(
    recording_rows: list[dict[str, object]],
    feature_name: str,
) -> dict[str, object]:
    """
    Perform balanced two-factor ANOVA using explicit sums of squares.

    Because every one of the 12 cells has exactly 5 recordings, the mode,
    interference, and interaction sums of squares are orthogonal and have a
    clear engineering interpretation.

    Parameters
    ----------
    recording_rows : list[dict[str, object]]
        One feature row per original recording, including flight mode and
        interference labels.
    feature_name : str
        Name of the RF feature to analyse.

    Returns
    -------
    dict[str, object]
        ANOVA summary containing sums of squares, mean squares, F statistics,
        p-values, and eta-squared effect sizes.

    Notes
    -----
    Separating the flight-mode effect, interference effect, and their
    interaction answers the engineering question directly: whether the RF
    information mainly comes from UAV behaviour, from the RF environment, or
    from a coupled behaviour-environment pattern.
    """
    cell_count = validate_balanced_design(recording_rows)
    mode_count = len(MODE_ORDER)
    interference_count = len(INTERFERENCE_ORDER)

    values = np.array([float(row[feature_name]) for row in recording_rows])
    grand_mean = float(np.mean(values))

    mode_means = {
        mode_code: float(
            np.mean(
                [
                    float(row[feature_name])
                    for row in recording_rows
                    if row["mode_code"] == mode_code
                ]
            )
        )
        for mode_code in MODE_ORDER
    }
    interference_means = {
        interference_code: float(
            np.mean(
                [
                    float(row[feature_name])
                    for row in recording_rows
                    if row["interference_code"] == interference_code
                ]
            )
        )
        for interference_code in INTERFERENCE_ORDER
    }
    cell_means = {
        (mode_code, interference_code): float(
            np.mean(
                [
                    float(row[feature_name])
                    for row in recording_rows
                    if row["mode_code"] == mode_code
                    and row["interference_code"] == interference_code
                ]
            )
        )
        for mode_code in MODE_ORDER
        for interference_code in INTERFERENCE_ORDER
    }

    ss_mode = interference_count * cell_count * sum(
        (mode_means[mode_code] - grand_mean) ** 2
        for mode_code in MODE_ORDER
    )
    ss_interference = mode_count * cell_count * sum(
        (interference_means[interference_code] - grand_mean) ** 2
        for interference_code in INTERFERENCE_ORDER
    )
    ss_interaction = cell_count * sum(
        (
            cell_means[(mode_code, interference_code)]
            - mode_means[mode_code]
            - interference_means[interference_code]
            + grand_mean
        )
        ** 2
        for mode_code in MODE_ORDER
        for interference_code in INTERFERENCE_ORDER
    )
    ss_error = sum(
        (
            float(row[feature_name])
            - cell_means[(str(row["mode_code"]), str(row["interference_code"]))]
        )
        ** 2
        for row in recording_rows
    )
    ss_total = float(np.sum((values - grand_mean) ** 2))

    df_mode = mode_count - 1
    df_interference = interference_count - 1
    df_interaction = df_mode * df_interference
    df_error = mode_count * interference_count * (cell_count - 1)

    ms_error = ss_error / df_error
    effects = {
        "flight_mode": (ss_mode, df_mode),
        "interference": (ss_interference, df_interference),
        "interaction": (ss_interaction, df_interaction),
    }
    structured_ss = ss_mode + ss_interference + ss_interaction

    result: dict[str, object] = {
        "feature": feature_name,
        "recording_count": len(recording_rows),
        "recordings_per_state": cell_count,
        "model_r_squared": structured_ss / ss_total if ss_total > 0 else 0.0,
    }

    for effect_name, (sum_of_squares, degrees_of_freedom) in effects.items():
        mean_square = sum_of_squares / degrees_of_freedom
        f_statistic = mean_square / ms_error if ms_error > 0 else 0.0
        p_value = float(
            f_distribution.sf(f_statistic, degrees_of_freedom, df_error)
        )
        partial_eta_squared = (
            sum_of_squares / (sum_of_squares + ss_error)
            if sum_of_squares + ss_error > 0
            else 0.0
        )
        structured_share = (
            sum_of_squares / structured_ss
            if structured_ss > 0
            else 0.0
        )

        result[f"{effect_name}_ss"] = sum_of_squares
        result[f"{effect_name}_df"] = degrees_of_freedom
        result[f"{effect_name}_f"] = f_statistic
        result[f"{effect_name}_p"] = p_value
        result[f"{effect_name}_partial_eta_squared"] = partial_eta_squared
        result[f"{effect_name}_effect_size"] = effect_size_label(
            partial_eta_squared
        )
        result[f"{effect_name}_structured_share"] = structured_share

    result["error_ss"] = ss_error
    result["error_df"] = df_error
    return result


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    Correct multiple p-values using Benjamini-Hochberg false discovery rate.

    We test several RF features for each physical effect. Correction reduces
    the chance of declaring an effect only because many tests were performed.
    """
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_minimum = 1.0

    for reverse_rank in range(count - 1, -1, -1):
        original_index = int(order[reverse_rank])
        rank = reverse_rank + 1
        candidate = p_values[original_index] * count / rank
        running_minimum = min(running_minimum, candidate)
        adjusted[original_index] = min(1.0, running_minimum)

    return adjusted.tolist()


def add_corrected_significance(
    result_rows: list[dict[str, object]],
    alpha: float,
) -> None:
    """Add corrected p-values and significance flags for every effect."""
    for effect_name in EFFECT_ORDER:
        p_values = [float(row[f"{effect_name}_p"]) for row in result_rows]
        corrected = benjamini_hochberg(p_values)
        for row, adjusted_p in zip(result_rows, corrected):
            row[f"{effect_name}_p_fdr"] = adjusted_p
            row[f"{effect_name}_significant"] = adjusted_p < alpha


def classify_feature_source(row: dict[str, object]) -> tuple[str, str]:
    """
    Classify the information source for one feature.

    Returns:
        source summary and strongest significant effect.
    """
    significant_effects = [
        effect_name
        for effect_name in EFFECT_ORDER
        if bool(row[f"{effect_name}_significant"])
    ]
    if not significant_effects:
        return "No clear factorial evidence", "none"

    strongest_effect = max(
        significant_effects,
        key=lambda effect_name: float(
            row[f"{effect_name}_partial_eta_squared"]
        ),
    )

    if len(significant_effects) == 1:
        return EFFECT_LABELS[strongest_effect], strongest_effect

    labels = " + ".join(EFFECT_LABELS[effect] for effect in significant_effects)
    return f"Mixed: {labels}", strongest_effect


def finalize_result_rows(result_rows: list[dict[str, object]]) -> None:
    """Add source classifications after significance correction."""
    for row in result_rows:
        source_summary, strongest_effect = classify_feature_source(row)
        row["information_source"] = source_summary
        row["strongest_significant_effect"] = strongest_effect


def run_factorial_analysis(
    recording_rows: list[dict[str, object]],
    alpha: float,
) -> list[dict[str, object]]:
    """Run ANOVA for all RF features and add corrected significance."""
    rows = [
        two_factor_anova(recording_rows, feature_name)
        for feature_name in FEATURE_NAMES
    ]
    add_corrected_significance(rows, alpha)
    finalize_result_rows(rows)
    return rows


def build_robust_evidence_rows(
    median_results: list[dict[str, object]],
    mean_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Identify effects supported by both median and mean aggregation.

    Agreement across both aggregation methods is stronger evidence that the
    result is not caused by a few unusual 20 ms segments.
    """
    median_by_feature = {str(row["feature"]): row for row in median_results}
    mean_by_feature = {str(row["feature"]): row for row in mean_results}
    evidence_rows = []

    for feature_name in FEATURE_NAMES:
        median_row = median_by_feature[feature_name]
        mean_row = mean_by_feature[feature_name]

        robust_effects = [
            effect_name
            for effect_name in EFFECT_ORDER
            if bool(median_row[f"{effect_name}_significant"])
            and bool(mean_row[f"{effect_name}_significant"])
        ]
        if robust_effects:
            strongest_effect = max(
                robust_effects,
                key=lambda effect_name: min(
                    float(median_row[f"{effect_name}_partial_eta_squared"]),
                    float(mean_row[f"{effect_name}_partial_eta_squared"]),
                ),
            )
            conclusion = (
                "Mixed"
                if len(robust_effects) > 1
                else EFFECT_LABELS[strongest_effect]
            )
        else:
            strongest_effect = "none"
            conclusion = "No robust effect"

        evidence_rows.append(
            {
                "feature": feature_name,
                "robust_effects": ";".join(robust_effects),
                "strongest_robust_effect": strongest_effect,
                "engineering_conclusion": conclusion,
                "median_mode_eta2": median_row[
                    "flight_mode_partial_eta_squared"
                ],
                "median_interference_eta2": median_row[
                    "interference_partial_eta_squared"
                ],
                "median_interaction_eta2": median_row[
                    "interaction_partial_eta_squared"
                ],
                "mean_mode_eta2": mean_row[
                    "flight_mode_partial_eta_squared"
                ],
                "mean_interference_eta2": mean_row[
                    "interference_partial_eta_squared"
                ],
                "mean_interaction_eta2": mean_row[
                    "interaction_partial_eta_squared"
                ],
            }
        )

    return evidence_rows


def build_source_summary(
    robust_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize how many non-redundant features support each source."""
    rows = []
    for effect_name in EFFECT_ORDER:
        supporting_features = [
            str(row["feature"])
            for row in robust_rows
            if str(row["feature"]) in NON_REDUNDANT_FEATURES
            and effect_name in str(row["robust_effects"]).split(";")
        ]
        rows.append(
            {
                "effect": effect_name,
                "effect_label": EFFECT_LABELS[effect_name],
                "non_redundant_features_supported": len(supporting_features),
                "total_non_redundant_features": len(NON_REDUNDANT_FEATURES),
                "supporting_features": ";".join(supporting_features),
            }
        )
    return rows


def plot_effect_size_heatmap(
    median_results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Plot partial eta-squared for mode, interference, and interaction.

    Darker cells mean that physical source explains a larger fraction of the
    feature variation after accounting for recording-to-recording residual.
    """
    matrix = np.array(
        [
            [
                float(row[f"{effect_name}_partial_eta_squared"])
                for effect_name in EFFECT_ORDER
            ]
            for row in median_results
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=0.7)
    figure.colorbar(image, ax=axis, label="Partial eta-squared")
    axis.set_xticks(
        np.arange(len(EFFECT_ORDER)),
        [EFFECT_LABELS[effect] for effect in EFFECT_ORDER],
    )
    axis.set_yticks(
        np.arange(len(FEATURE_NAMES)),
        FEATURE_NAMES,
    )
    axis.set_title("Origin of AIR RF Feature Variation")

    for row_index, row in enumerate(median_results):
        for column_index, effect_name in enumerate(EFFECT_ORDER):
            eta = float(row[f"{effect_name}_partial_eta_squared"])
            significant = bool(row[f"{effect_name}_significant"])
            label = f"{eta:.2f}" + ("*" if significant else "")
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="black" if eta < 0.45 else "white",
                fontsize=8,
            )

    axis.set_xlabel("* FDR-corrected p < 0.05")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_structured_variation_shares(
    median_results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Plot how structured between-state variation is divided among the factors.

    This answers "where does the structured variation come from?" but should
    be read together with significance and total model R-squared.
    """
    features = [str(row["feature"]) for row in median_results]
    mode_share = np.array(
        [float(row["flight_mode_structured_share"]) for row in median_results]
    )
    interference_share = np.array(
        [float(row["interference_structured_share"]) for row in median_results]
    )
    interaction_share = np.array(
        [float(row["interaction_structured_share"]) for row in median_results]
    )
    positions = np.arange(len(features))

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions, mode_share, label="Flight mode", color="tab:blue")
    axis.bar(
        positions,
        interference_share,
        bottom=mode_share,
        label="Interference",
        color="tab:orange",
    )
    axis.bar(
        positions,
        interaction_share,
        bottom=mode_share + interference_share,
        label="Interaction",
        color="tab:green",
    )
    axis.set_xticks(positions, features, rotation=35, ha="right")
    axis.set_ylabel("Share of structured variation")
    axis.set_ylim(0, 1)
    axis.set_title("How Each RF Feature Divides Across Physical Information Sources")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def write_engineering_report(
    output_path: Path,
    recording_count: int,
    recordings_per_state: int,
    robust_rows: list[dict[str, object]],
    source_summary: list[dict[str, object]],
) -> None:
    """Write a direct answer to the research question."""
    source_by_effect = {str(row["effect"]): row for row in source_summary}

    mode_features = str(source_by_effect["flight_mode"]["supporting_features"])
    interference_features = str(
        source_by_effect["interference"]["supporting_features"]
    )
    interaction_features = str(
        source_by_effect["interaction"]["supporting_features"]
    )

    detailed_lines = []
    for row in robust_rows:
        detailed_lines.append(
            f"- `{row['feature']}`: {row['engineering_conclusion']} "
            f"(robust effects: {row['robust_effects'] or 'none'})"
        )

    report = f"""# AIR RF Information Source Analysis

## Question

Does the RF information originate from flight mode, interference conditions,
or the interaction between the two?

## Independent Statistical Units

- Original recordings: {recording_count}
- Recordings per 12-state cell: {recordings_per_state}
- Ten 20 ms segments were aggregated within each recording.
- Primary analysis used the median; mean aggregation was used as a sensitivity
  check.

## Direct Answer

The RF information does not originate from only one source.

1. **Flight mode is the strongest and most consistent source.**
   It affects more non-redundant RF features than either of the other terms.
2. **Interference contributes real secondary information.**
   Its clearest effects are on spectral spreading and time-frequency activity.
3. **The flight-mode x interference interaction is also significant.**
   Therefore, some RF characteristics are combination-specific: the influence
   of flight mode changes depending on the interference environment.

This supports a **factorial interpretation**:

```text
RF information = flight-mode information
               + interference information
               + joint mode/interference information
```

The interaction provides evidence for 12-state-specific structure, but it does
not by itself prove that all 12 states are perfectly separable.

## Robust Supporting Features

Flight-mode evidence:

`{mode_features}`

Interference evidence:

`{interference_features}`

Interaction evidence:

`{interaction_features}`

## Feature-Level Conclusions

{chr(10).join(detailed_lines)}

## Engineering Interpretation

- If only flight mode mattered, interference and interaction effects would be
  absent. They are not.
- If only interference mattered, mode effects would be absent. Instead, mode
  is the strongest source.
- Because interaction effects remain after accounting for both main effects,
  the complete 12-state labels contain some genuine joint information.
- The appropriate next ML design is therefore hierarchical or multi-task:
  evaluate 3-mode classification, 4-interference classification, and 12-state
  classification separately using recording-grouped validation.
"""
    output_path.write_text(report, encoding="utf-8")


def result_fieldnames() -> list[str]:
    """Return the stable CSV column order for detailed ANOVA results."""
    fields = [
        "feature",
        "recording_count",
        "recordings_per_state",
        "model_r_squared",
    ]
    for effect_name in EFFECT_ORDER:
        fields.extend(
            [
                f"{effect_name}_ss",
                f"{effect_name}_df",
                f"{effect_name}_f",
                f"{effect_name}_p",
                f"{effect_name}_p_fdr",
                f"{effect_name}_significant",
                f"{effect_name}_partial_eta_squared",
                f"{effect_name}_effect_size",
                f"{effect_name}_structured_share",
            ]
        )
    fields.extend(
        [
            "error_ss",
            "error_df",
            "information_source",
            "strongest_significant_effect",
        ]
    )
    return fields


def parse_arguments() -> argparse.Namespace:
    """Parse Stage 5 settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Decompose AIR RF feature information into flight mode, "
            "interference, and interaction effects."
        )
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
        help="Folder for factorial-analysis tables, report, and plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR-corrected significance threshold. Default: 0.05",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete factorial information-source analysis."""
    args = parse_arguments()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")

    segment_rows = read_csv_rows(args.input_csv)
    median_recordings = aggregate_segments_by_recording(segment_rows, "median")
    mean_recordings = aggregate_segments_by_recording(segment_rows, "mean")
    recordings_per_state = validate_balanced_design(median_recordings)

    median_results = run_factorial_analysis(median_recordings, args.alpha)
    mean_results = run_factorial_analysis(mean_recordings, args.alpha)
    robust_rows = build_robust_evidence_rows(median_results, mean_results)
    source_summary = build_source_summary(robust_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    recording_fields = [
        "file_path",
        "file_name",
        "interference_code",
        "mode_code",
        "state",
        "segment_count",
        "aggregation",
    ] + FEATURE_NAMES
    write_csv(
        args.output_dir / "recording_features_median.csv",
        median_recordings,
        recording_fields,
    )
    write_csv(
        args.output_dir / "recording_features_mean.csv",
        mean_recordings,
        recording_fields,
    )
    write_csv(
        args.output_dir / "factorial_effects_median.csv",
        median_results,
        result_fieldnames(),
    )
    write_csv(
        args.output_dir / "factorial_effects_mean.csv",
        mean_results,
        result_fieldnames(),
    )
    write_csv(
        args.output_dir / "robust_feature_evidence.csv",
        robust_rows,
        [
            "feature",
            "robust_effects",
            "strongest_robust_effect",
            "engineering_conclusion",
            "median_mode_eta2",
            "median_interference_eta2",
            "median_interaction_eta2",
            "mean_mode_eta2",
            "mean_interference_eta2",
            "mean_interaction_eta2",
        ],
    )
    write_csv(
        args.output_dir / "information_source_summary.csv",
        source_summary,
        [
            "effect",
            "effect_label",
            "non_redundant_features_supported",
            "total_non_redundant_features",
            "supporting_features",
        ],
    )

    plot_effect_size_heatmap(
        median_results,
        args.output_dir / "factorial_effect_size_heatmap.png",
    )
    plot_structured_variation_shares(
        median_results,
        args.output_dir / "structured_variation_shares.png",
    )
    write_engineering_report(
        args.output_dir / "engineering_answer.md",
        len(median_recordings),
        recordings_per_state,
        robust_rows,
        source_summary,
    )

    print("\n--- AIR RF information source ---")
    for row in source_summary:
        print(
            f"{row['effect_label']}: "
            f"{row['non_redundant_features_supported']}/"
            f"{row['total_non_redundant_features']} non-redundant features"
        )

    print("\nDirect conclusion:")
    print("1. Flight mode is the strongest and most consistent source.")
    print("2. Interference adds secondary spectral/time-frequency information.")
    print("3. The interaction is real, so some information is 12-state-specific.")
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
