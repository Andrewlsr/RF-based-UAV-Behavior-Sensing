"""
Stage 4: Compare AIR drone RF states before machine learning.

Research question:
    Are the 12 AIR operating states separable based on RF characteristics in
    the frequency domain and time-frequency domain?

The 12 states are:
    4 interference conditions x 3 flight modes

Interference conditions:
    00 = clean
    01 = Bluetooth
    10 = Wi-Fi
    11 = Bluetooth + Wi-Fi

Flight modes:
    ON = switched on
    HO = hovering
    FY = flying

Engineering approach:
    - Process every AIR .dat file.
    - Read a fixed-length IQ segment from each file, default 20 ms.
    - Compute FFT, PSD, and spectrogram.
    - Extract interpretable RF features.
    - Aggregate feature values into summary tables for all 12 states.
    - Estimate whether state clusters look separable before ML.

This script intentionally does not train a model. It prepares evidence for
whether traditional ML is likely to have useful signal structure to learn from.
"""

from pathlib import Path

import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft
from scipy import signal

from stage1_load_air_iq import (
    DEFAULT_CENTER_FREQUENCY_HZ,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DC_EXCLUSION_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    FLOAT32_BYTES,
    FLIGHT_MODE_LABELS,
    INTERFERENCE_LABELS,
    count_raw_floats_from_file_size,
    find_air_dat_files,
    parse_recording_metadata,
)


DEFAULT_SEGMENT_MS = 20.0
DEFAULT_START_MS = 0.0
DEFAULT_SEGMENTS_PER_FILE = 10
DEFAULT_FFT_SAMPLES = 65_536
DEFAULT_PSD_NPERSEG = 8192
DEFAULT_PSD_OVERLAP = 4096
DEFAULT_SPECTROGRAM_NPERSEG = 4096
DEFAULT_SPECTROGRAM_OVERLAP = 2048
DEFAULT_OUTPUT_DIR = Path("outputs") / "stage4_air_state_analysis"


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


INTERFERENCE_ORDER = ["00", "01", "10", "11"]
MODE_ORDER = ["ON", "HO", "FY"]


def load_iq_segment(
    file_path: Path,
    start_sample: int,
    sample_count: int,
) -> np.ndarray:
    """
    Load one fixed-length complex IQ segment from a DroneDetect .dat file.

    The .dat format is interleaved float32:
        I0, Q0, I1, Q1, ...

    A 20 ms segment at 60 MHz is 1,200,000 complex IQ samples. Reading only
    the segment keeps the analysis memory-safe while still processing every
    AIR recording.
    """
    raw_start_float = 2 * start_sample
    raw_float_count = 2 * sample_count
    byte_offset = raw_start_float * FLOAT32_BYTES

    with file_path.open("rb") as file_handle:
        file_handle.seek(byte_offset)
        raw = np.fromfile(file_handle, dtype=np.float32, count=raw_float_count)

    if raw.size < 4:
        raise ValueError(f"Not enough IQ data could be read from {file_path}")
    if raw.size % 2 != 0:
        raw = raw[:-1]

    i_channel = raw[0::2]
    q_channel = raw[1::2]
    return i_channel + 1j * q_channel


def choose_evenly_spaced_segment_starts(
    total_iq_samples: int,
    segment_samples: int,
    first_start_sample: int,
    segments_per_file: int,
) -> list[int]:
    """
    Choose segment start positions across the recording.

    Engineering meaning:
        Using only the first 20 ms can be misleading. Evenly spaced segments
        sample the whole recording, so the feature matrix better represents
        stable RF behaviour instead of one arbitrary moment.
    """
    if first_start_sample >= total_iq_samples:
        raise ValueError("First segment start is beyond the end of the recording.")

    last_valid_start = max(first_start_sample, total_iq_samples - segment_samples)
    if segments_per_file == 1 or last_valid_start == first_start_sample:
        return [first_start_sample]

    starts = np.linspace(
        first_start_sample,
        last_valid_start,
        num=segments_per_file,
    )
    return [int(round(start)) for start in starts]


def remove_dc(iq_signal: np.ndarray) -> np.ndarray:
    """
    Remove receiver DC offset from the IQ segment.

    RF meaning:
        SDRs often produce artificial energy at the exact centre frequency.
        Subtracting the mean reduces that hardware artifact before spectral
        features are computed.
    """
    return iq_signal - np.mean(iq_signal)


def compute_fft_power(
    iq_signal: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one FFT power spectrum for a segment.

    FFT gives a single frequency-domain snapshot of the selected IQ block.
    """
    iq_without_dc = remove_dc(iq_signal)
    fft_sample_count = min(DEFAULT_FFT_SAMPLES, iq_without_dc.size)
    fft_block = iq_without_dc[:fft_sample_count]
    window = np.hanning(fft_block.size)

    fft_values = scipy_fft.fftshift(scipy_fft.fft(fft_block * window))
    frequency_hz = scipy_fft.fftshift(
        scipy_fft.fftfreq(fft_block.size, d=1.0 / sample_rate_hz)
    )

    power = np.abs(fft_values) ** 2
    return frequency_hz, power


def compute_psd(
    iq_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for a segment.

    PSD is more stable than a single FFT because it averages spectra over
    multiple windows. This makes it useful for comparing recordings.
    """
    iq_without_dc = remove_dc(iq_signal)
    nperseg = min(nperseg, iq_without_dc.size)
    noverlap = min(noverlap, nperseg - 1)

    frequency_hz, psd = signal.welch(
        iq_without_dc,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling="density",
    )

    return scipy_fft.fftshift(frequency_hz), scipy_fft.fftshift(psd)


def compute_spectrogram_power(
    iq_signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a two-sided spectrogram for complex IQ.

    The spectrogram captures time-frequency behaviour: bursts, hopping,
    persistent channels, and changing RF activity over the segment.
    """
    iq_without_dc = remove_dc(iq_signal)
    nperseg = min(nperseg, iq_without_dc.size)
    noverlap = min(noverlap, nperseg - 1)

    frequency_hz, time_s, spectrogram_power = signal.spectrogram(
        iq_without_dc,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )

    return (
        scipy_fft.fftshift(frequency_hz),
        time_s,
        scipy_fft.fftshift(spectrogram_power, axes=0),
    )


def valid_frequency_mask(frequency_hz: np.ndarray, dc_exclusion_hz: float) -> np.ndarray:
    """Exclude the centre/DC region from features that should describe RF content."""
    return np.abs(frequency_hz) > dc_exclusion_hz


def spectral_entropy(power: np.ndarray) -> float:
    """
    Normalized spectral entropy.

    Low entropy means energy is concentrated in a few frequencies.
    High entropy means energy is spread more evenly across the spectrum.
    """
    power = np.maximum(np.asarray(power, dtype=float), 0.0)
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0

    probabilities = power / total_power
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / math.log2(power.size))


def occupied_bandwidth_hz(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    fraction: float = 0.99,
) -> float:
    """
    Estimate occupied bandwidth containing a given fraction of total power.

    For example, 99% occupied bandwidth is the frequency span that contains
    the central 99% of spectral energy.
    """
    order = np.argsort(frequency_hz)
    sorted_frequency = frequency_hz[order]
    sorted_power = np.maximum(power[order], 0.0)

    cumulative_power = np.cumsum(sorted_power)
    total_power = cumulative_power[-1]
    if total_power <= 0:
        return 0.0

    lower_target = (1.0 - fraction) / 2.0 * total_power
    upper_target = (1.0 + fraction) / 2.0 * total_power
    lower_index = int(np.searchsorted(cumulative_power, lower_target))
    upper_index = int(np.searchsorted(cumulative_power, upper_target))
    lower_index = min(lower_index, sorted_frequency.size - 1)
    upper_index = min(upper_index, sorted_frequency.size - 1)

    return float(sorted_frequency[upper_index] - sorted_frequency[lower_index])


def extract_features(
    iq_signal: np.ndarray,
    sample_rate_hz: float,
    psd_nperseg: int,
    psd_overlap: int,
    spectrogram_nperseg: int,
    spectrogram_overlap: int,
    dc_exclusion_hz: float,
) -> dict[str, float]:
    """
    Extract interpretable RF features from one recording segment.

    Features are deliberately engineering-oriented rather than black-box:
        - power/energy features describe signal strength
        - peak/centroid/bandwidth describe where frequency energy lives
        - entropy describes concentration versus spread
        - spectrogram features describe time variation and activity fraction

    Parameters
    ----------
    iq_signal : np.ndarray
        Complex IQ samples for one selected analysis segment.
    sample_rate_hz : float
        Sampling rate of the recording. DroneDetect AIR recordings use
        60 MHz, which defines the FFT, PSD, and spectrogram frequency axes.
    psd_nperseg, psd_overlap : int
        Welch PSD window length and overlap.
    spectrogram_nperseg, spectrogram_overlap : int
        Short-time Fourier transform window length and overlap used for the
        time-frequency features.
    dc_exclusion_hz : float
        Frequency region around 0 Hz ignored when selecting non-centre RF
        activity, because SDR recordings often contain a centre/DC artifact.

    Returns
    -------
    dict[str, float]
        Feature name to numeric value for downstream separability analysis.

    Notes
    -----
    Stage 4 is still a research analysis step: it asks whether simple,
    explainable RF quantities show state-dependent structure before any
    machine-learning model is trusted.
    """
    iq_without_dc = remove_dc(iq_signal)
    time_power = np.abs(iq_without_dc) ** 2

    fft_frequency_hz, fft_power = compute_fft_power(iq_signal, sample_rate_hz)
    psd_frequency_hz, psd_power = compute_psd(
        iq_signal,
        sample_rate_hz,
        psd_nperseg,
        psd_overlap,
    )
    _, _, spectrogram_power = compute_spectrogram_power(
        iq_signal,
        sample_rate_hz,
        spectrogram_nperseg,
        spectrogram_overlap,
    )

    fft_mask = valid_frequency_mask(fft_frequency_hz, dc_exclusion_hz)
    psd_mask = valid_frequency_mask(psd_frequency_hz, dc_exclusion_hz)

    fft_peak_index = np.where(fft_mask)[0][int(np.argmax(fft_power[fft_mask]))]
    psd_peak_index = np.where(psd_mask)[0][int(np.argmax(psd_power[psd_mask]))]

    valid_psd_frequency = psd_frequency_hz[psd_mask]
    valid_psd_power = np.maximum(psd_power[psd_mask], 0.0)
    total_psd_power = np.sum(valid_psd_power)

    if total_psd_power <= 0:
        spectral_centroid = 0.0
    else:
        spectral_centroid = float(
            np.sum(valid_psd_frequency * valid_psd_power) / total_psd_power
        )

    spectrogram_column_power = np.sum(spectrogram_power, axis=0)
    mean_column_power = np.mean(spectrogram_column_power)
    if mean_column_power <= 0:
        temporal_variability = 0.0
    else:
        temporal_variability = float(np.std(spectrogram_column_power) / mean_column_power)

    spectrogram_peak = np.max(spectrogram_power)
    if spectrogram_peak <= 0:
        active_fraction = 0.0
    else:
        spectrogram_db = 10.0 * np.log10(
            np.maximum(spectrogram_power / spectrogram_peak, 1e-12)
        )
        active_fraction = float(np.mean(spectrogram_db > -50.0))

    return {
        "rms_power": float(np.sqrt(np.mean(time_power))),
        "signal_energy": float(np.sum(time_power)),
        "fft_peak_frequency_mhz": float(fft_frequency_hz[fft_peak_index] / 1e6),
        "peak_frequency_mhz": float(psd_frequency_hz[psd_peak_index] / 1e6),
        "occupied_bandwidth_mhz": occupied_bandwidth_hz(
            valid_psd_frequency,
            valid_psd_power,
        )
        / 1e6,
        "spectral_entropy": spectral_entropy(valid_psd_power),
        "spectral_centroid_mhz": spectral_centroid / 1e6,
        "spectrogram_temporal_variability": temporal_variability,
        "spectrogram_active_fraction": active_fraction,
    }


def state_id(interference_code: str, mode_code: str) -> str:
    """Compact label for one of the 12 AIR operating states."""
    return f"{interference_code}_{mode_code}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_records(
    records: list[dict[str, object]],
    keys: tuple[str, ...],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    """Group record dictionaries by one or more metadata keys."""
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in records:
        group_key = tuple(record[key] for key in keys)
        grouped.setdefault(group_key, []).append(record)
    return grouped


def make_state_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create mean/std feature summary for each of the 12 AIR states."""
    grouped = group_records(records, ("interference_code", "mode_code"))
    summary_rows = []

    for interference_code in INTERFERENCE_ORDER:
        for mode_code in MODE_ORDER:
            state_records = grouped.get((interference_code, mode_code), [])
            if not state_records:
                continue

            row: dict[str, object] = {
                "state": state_id(interference_code, mode_code),
                "interference_code": interference_code,
                "interference_label": INTERFERENCE_LABELS[interference_code],
                "mode_code": mode_code,
                "mode_label": FLIGHT_MODE_LABELS[mode_code],
                "source_recording_count": len(
                    {str(record["file_path"]) for record in state_records}
                ),
                "segment_count": len(state_records),
            }

            for feature_name in FEATURE_NAMES:
                values = np.array([float(record[feature_name]) for record in state_records])
                row[f"{feature_name}_mean"] = float(np.mean(values))
                row[f"{feature_name}_std"] = float(np.std(values, ddof=0))

            summary_rows.append(row)

    return summary_rows


def make_analysis_a_rows(state_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Analysis A:
        Fix flight mode and compare the four interference conditions.
    """
    by_state = {
        (row["interference_code"], row["mode_code"]): row
        for row in state_summary
    }
    rows = []

    for mode_code in MODE_ORDER:
        for feature_name in FEATURE_NAMES:
            row: dict[str, object] = {
                "analysis": "A_fix_mode_compare_interference",
                "mode_code": mode_code,
                "mode_label": FLIGHT_MODE_LABELS[mode_code],
                "feature": feature_name,
            }
            values = []
            for interference_code in INTERFERENCE_ORDER:
                state_row = by_state.get((interference_code, mode_code))
                if state_row is None:
                    row[f"{interference_code}_mean"] = ""
                    continue
                value = state_row[f"{feature_name}_mean"]
                row[f"{interference_code}_mean"] = value
                values.append(float(value))

            if values:
                row["range"] = max(values) - min(values)
                row["max_condition"] = INTERFERENCE_ORDER[int(np.argmax(values))]
                row["min_condition"] = INTERFERENCE_ORDER[int(np.argmin(values))]
            else:
                row["range"] = ""
                row["max_condition"] = ""
                row["min_condition"] = ""
            rows.append(row)

    return rows


def make_analysis_b_rows(state_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Analysis B:
        Fix interference condition and compare the three flight modes.
    """
    by_state = {
        (row["interference_code"], row["mode_code"]): row
        for row in state_summary
    }
    rows = []

    for interference_code in INTERFERENCE_ORDER:
        for feature_name in FEATURE_NAMES:
            row: dict[str, object] = {
                "analysis": "B_fix_interference_compare_mode",
                "interference_code": interference_code,
                "interference_label": INTERFERENCE_LABELS[interference_code],
                "feature": feature_name,
            }
            values = []
            for mode_code in MODE_ORDER:
                state_row = by_state.get((interference_code, mode_code))
                if state_row is None:
                    row[f"{mode_code}_mean"] = ""
                    continue
                value = state_row[f"{feature_name}_mean"]
                row[f"{mode_code}_mean"] = value
                values.append(float(value))

            if values:
                row["range"] = max(values) - min(values)
                row["max_mode"] = MODE_ORDER[int(np.argmax(values))]
                row["min_mode"] = MODE_ORDER[int(np.argmin(values))]
            else:
                row["range"] = ""
                row["max_mode"] = ""
                row["min_mode"] = ""
            rows.append(row)

    return rows


def rank_feature_variation(
    records: list[dict[str, object]],
    group_key: str,
) -> list[dict[str, object]]:
    """
    Rank features by how much their group means vary.

    The normalized score is:
        range(group means) / overall standard deviation

    Higher score means a feature changes more strongly across the chosen
    grouping, making it more promising for separability.
    """
    grouped = group_records(records, (group_key,))
    rows = []

    for feature_name in FEATURE_NAMES:
        all_values = np.array([float(record[feature_name]) for record in records])
        overall_std = float(np.std(all_values, ddof=0))
        group_means = []

        for group_value, group_records_list in grouped.items():
            values = np.array(
                [float(record[feature_name]) for record in group_records_list]
            )
            group_means.append(float(np.mean(values)))

        feature_range = max(group_means) - min(group_means)
        normalized_score = feature_range / (overall_std + 1e-12)
        rows.append(
            {
                "group_key": group_key,
                "feature": feature_name,
                "group_mean_range": feature_range,
                "overall_std": overall_std,
                "normalized_variation_score": normalized_score,
            }
        )

    return sorted(rows, key=lambda row: row["normalized_variation_score"], reverse=True)


def estimate_state_separability(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Estimate pre-ML class separability using standardized feature centroids.

    For each state:
        - compute its centroid in standardized feature space
        - find the nearest other state centroid
        - compare that distance with the state's within-state spread

    This is not a classifier. It is an engineering separability check.
    """
    feature_matrix = np.array(
        [[float(record[feature_name]) for feature_name in FEATURE_NAMES] for record in records]
    )
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_std = np.std(feature_matrix, axis=0)
    feature_std[feature_std == 0] = 1.0
    standardized = (feature_matrix - feature_mean) / feature_std

    state_labels = [
        state_id(str(record["interference_code"]), str(record["mode_code"]))
        for record in records
    ]
    unique_states = [
        state_id(interference_code, mode_code)
        for interference_code in INTERFERENCE_ORDER
        for mode_code in MODE_ORDER
    ]

    centroids: dict[str, np.ndarray] = {}
    within_radius: dict[str, float] = {}

    for state_label in unique_states:
        indices = [index for index, label in enumerate(state_labels) if label == state_label]
        if not indices:
            continue

        state_points = standardized[indices, :]
        centroid = np.mean(state_points, axis=0)
        centroids[state_label] = centroid
        distances = np.linalg.norm(state_points - centroid, axis=1)
        within_radius[state_label] = float(np.mean(distances))

    rows = []
    for state_label in unique_states:
        if state_label not in centroids:
            continue

        other_states = [label for label in centroids if label != state_label]
        if not other_states:
            rows.append(
                {
                    "state": state_label,
                    "nearest_state": "",
                    "nearest_centroid_distance": "",
                    "within_state_radius": within_radius[state_label],
                    "separation_ratio": "",
                    "pre_ml_separability": "not_enough_states",
                }
            )
            continue

        distances_to_other_states = [
            float(np.linalg.norm(centroids[state_label] - centroids[other_label]))
            for other_label in other_states
        ]
        nearest_index = int(np.argmin(distances_to_other_states))
        nearest_state = other_states[nearest_index]
        nearest_distance = distances_to_other_states[nearest_index]
        radius = within_radius[state_label]
        separation_ratio = nearest_distance / (radius + 1e-12)

        if separation_ratio >= 2.0:
            separability_label = "strong"
        elif separation_ratio >= 1.0:
            separability_label = "moderate"
        else:
            separability_label = "weak"

        rows.append(
            {
                "state": state_label,
                "nearest_state": nearest_state,
                "nearest_centroid_distance": nearest_distance,
                "within_state_radius": radius,
                "separation_ratio": separation_ratio,
                "pre_ml_separability": separability_label,
            }
        )

    return rows


def standardize_feature_matrix(
    records: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert feature records to a standardized numeric matrix.

    Standardization puts all features on comparable scale. Without it, a large
    unit such as signal energy would dominate features with smaller numeric
    ranges such as spectral entropy.
    """
    feature_matrix = np.array(
        [[float(record[feature_name]) for feature_name in FEATURE_NAMES] for record in records]
    )
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_std = np.std(feature_matrix, axis=0)
    feature_std[feature_std == 0] = 1.0
    return (feature_matrix - feature_mean) / feature_std, feature_mean, feature_std


def compute_pca_scores(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D PCA projection using NumPy SVD.

    PCA here is not a classifier. It is an engineering visualization: if states
    occupy visibly different regions in the first two principal components,
    the feature set contains separable structure.
    """
    standardized, _, _ = standardize_feature_matrix(records)
    _, singular_values, components_t = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ components_t[:2].T
    explained_variance = singular_values**2 / (standardized.shape[0] - 1)
    explained_ratio = explained_variance / np.sum(explained_variance)
    return scores, explained_ratio[:2]


def make_feature_interval_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Build min/mean/max intervals for every state and feature.

    Engineers can read this directly: if two states have non-overlapping
    feature ranges, that feature gives direct separability evidence.
    """
    grouped = group_records(records, ("interference_code", "mode_code"))
    rows = []

    for interference_code in INTERFERENCE_ORDER:
        for mode_code in MODE_ORDER:
            state_records = grouped.get((interference_code, mode_code), [])
            if not state_records:
                continue

            state_label = state_id(interference_code, mode_code)
            for feature_name in FEATURE_NAMES:
                values = np.array([float(record[feature_name]) for record in state_records])
                rows.append(
                    {
                        "state": state_label,
                        "interference_code": interference_code,
                        "mode_code": mode_code,
                        "feature": feature_name,
                        "min": float(np.min(values)),
                        "mean": float(np.mean(values)),
                        "max": float(np.max(values)),
                        "std": float(np.std(values, ddof=0)),
                    }
                )

    return rows


def intervals_do_not_overlap(
    left_min: float,
    left_max: float,
    right_min: float,
    right_max: float,
) -> bool:
    """Return True when two numeric intervals are clearly separated."""
    return left_max < right_min or right_max < left_min


def make_pairwise_engineering_separability(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """
    Compare every pair of states using simple feature range overlap.

    Plain-language criterion:
        - STRONG: 3 or more features have non-overlapping min-max ranges
        - PARTIAL: 1 or 2 features have non-overlapping min-max ranges
        - OVERLAP: no current feature fully separates the pair

    This is intentionally simple and inspectable. It is not a model.
    """
    interval_rows = make_feature_interval_rows(records)
    intervals = {
        (row["state"], row["feature"]): row
        for row in interval_rows
    }
    states = [
        state_id(interference_code, mode_code)
        for interference_code in INTERFERENCE_ORDER
        for mode_code in MODE_ORDER
    ]
    states = [state for state in states if any(row["state"] == state for row in interval_rows)]

    pair_rows = []
    feature_separation_counts = {feature_name: 0 for feature_name in FEATURE_NAMES}

    for left_index, left_state in enumerate(states):
        for right_state in states[left_index + 1 :]:
            separating_features = []

            for feature_name in FEATURE_NAMES:
                left_interval = intervals[(left_state, feature_name)]
                right_interval = intervals[(right_state, feature_name)]
                if intervals_do_not_overlap(
                    float(left_interval["min"]),
                    float(left_interval["max"]),
                    float(right_interval["min"]),
                    float(right_interval["max"]),
                ):
                    separating_features.append(feature_name)
                    feature_separation_counts[feature_name] += 1

            if len(separating_features) >= 3:
                evidence = "STRONG"
            elif separating_features:
                evidence = "PARTIAL"
            else:
                evidence = "OVERLAP"

            pair_rows.append(
                {
                    "state_a": left_state,
                    "state_b": right_state,
                    "non_overlapping_feature_count": len(separating_features),
                    "separating_features": ";".join(separating_features),
                    "engineering_evidence": evidence,
                }
            )

    feature_rows = []
    total_pairs = len(pair_rows)
    for feature_name in FEATURE_NAMES:
        count = feature_separation_counts[feature_name]
        feature_rows.append(
            {
                "feature": feature_name,
                "state_pairs_separated": count,
                "total_state_pairs": total_pairs,
                "pair_separation_percent": 100.0 * count / total_pairs if total_pairs else 0.0,
            }
        )
    feature_rows.sort(key=lambda row: row["state_pairs_separated"], reverse=True)

    strong_count = sum(row["engineering_evidence"] == "STRONG" for row in pair_rows)
    partial_count = sum(row["engineering_evidence"] == "PARTIAL" for row in pair_rows)
    overlap_count = sum(row["engineering_evidence"] == "OVERLAP" for row in pair_rows)
    summary = {
        "total_state_pairs": total_pairs,
        "strong_pairs": strong_count,
        "partial_pairs": partial_count,
        "overlap_pairs": overlap_count,
        "strong_or_partial_pairs": strong_count + partial_count,
        "strong_or_partial_percent": 100.0 * (strong_count + partial_count) / total_pairs
        if total_pairs
        else 0.0,
    }

    return pair_rows, feature_rows, summary


def make_pairwise_central_interval_separability(
    records: list[dict[str, object]],
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """
    Compare states using central percentile intervals instead of min-max ranges.

    Why this is useful:
        With 10 segments per file, each state has many more samples. A single
        unusual segment can make min-max intervals overlap even when the state
        distributions are mostly different. The central 80% interval gives a
        more practical engineering view of typical behaviour.
    """
    states = [
        state_id(interference_code, mode_code)
        for interference_code in INTERFERENCE_ORDER
        for mode_code in MODE_ORDER
    ]
    states = [state for state in states if any(str(record["state"]) == state for record in records)]

    feature_intervals: dict[tuple[str, str], tuple[float, float]] = {}
    for state_label in states:
        state_records = [record for record in records if str(record["state"]) == state_label]
        for feature_name in FEATURE_NAMES:
            values = np.array([float(record[feature_name]) for record in state_records])
            lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
            feature_intervals[(state_label, feature_name)] = (float(lower), float(upper))

    pair_rows = []
    feature_separation_counts = {feature_name: 0 for feature_name in FEATURE_NAMES}

    for left_index, left_state in enumerate(states):
        for right_state in states[left_index + 1 :]:
            separating_features = []

            for feature_name in FEATURE_NAMES:
                left_lower, left_upper = feature_intervals[(left_state, feature_name)]
                right_lower, right_upper = feature_intervals[(right_state, feature_name)]
                if intervals_do_not_overlap(left_lower, left_upper, right_lower, right_upper):
                    separating_features.append(feature_name)
                    feature_separation_counts[feature_name] += 1

            if len(separating_features) >= 3:
                evidence = "STRONG"
            elif separating_features:
                evidence = "PARTIAL"
            else:
                evidence = "OVERLAP"

            pair_rows.append(
                {
                    "state_a": left_state,
                    "state_b": right_state,
                    "central_interval": f"{lower_percentile:.0f}-{upper_percentile:.0f} percentile",
                    "non_overlapping_feature_count": len(separating_features),
                    "separating_features": ";".join(separating_features),
                    "engineering_evidence": evidence,
                }
            )

    total_pairs = len(pair_rows)
    feature_rows = []
    for feature_name in FEATURE_NAMES:
        count = feature_separation_counts[feature_name]
        feature_rows.append(
            {
                "feature": feature_name,
                "state_pairs_separated": count,
                "total_state_pairs": total_pairs,
                "pair_separation_percent": 100.0 * count / total_pairs if total_pairs else 0.0,
            }
        )
    feature_rows.sort(key=lambda row: row["state_pairs_separated"], reverse=True)

    strong_count = sum(row["engineering_evidence"] == "STRONG" for row in pair_rows)
    partial_count = sum(row["engineering_evidence"] == "PARTIAL" for row in pair_rows)
    overlap_count = sum(row["engineering_evidence"] == "OVERLAP" for row in pair_rows)
    summary = {
        "total_state_pairs": total_pairs,
        "strong_pairs": strong_count,
        "partial_pairs": partial_count,
        "overlap_pairs": overlap_count,
        "strong_or_partial_pairs": strong_count + partial_count,
        "strong_or_partial_percent": 100.0 * (strong_count + partial_count) / total_pairs
        if total_pairs
        else 0.0,
    }

    return pair_rows, feature_rows, summary


def plot_state_feature_heatmap(
    state_summary: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save a heatmap of standardized state-level feature means."""
    state_labels = [str(row["state"]) for row in state_summary]
    matrix = np.array(
        [
            [float(row[f"{feature_name}_mean"]) for feature_name in FEATURE_NAMES]
            for row in state_summary
        ]
    )

    column_mean = np.mean(matrix, axis=0)
    column_std = np.std(matrix, axis=0)
    column_std[column_std == 0] = 1.0
    z_matrix = (matrix - column_mean) / column_std

    plt.figure(figsize=(13, 7))
    image = plt.imshow(z_matrix, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    plt.colorbar(image, label="State mean feature value (z-score)")
    plt.xticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=35, ha="right")
    plt.yticks(np.arange(len(state_labels)), state_labels)
    plt.title("AIR State Feature Summary Before Machine Learning")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pca_separability(
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Save a PCA scatter plot coloured by the 12 AIR states.

    This is the quickest visual answer to the separability question: separated
    clouds suggest useful RF feature structure; heavy overlap suggests the
    current feature set is not enough yet.
    """
    scores, explained_ratio = compute_pca_scores(records)
    states = [str(record["state"]) for record in records]
    unique_states = [
        state_id(interference_code, mode_code)
        for interference_code in INTERFERENCE_ORDER
        for mode_code in MODE_ORDER
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_states)))

    plt.figure(figsize=(10, 8))
    for state_label, color in zip(unique_states, colors):
        indices = [index for index, state in enumerate(states) if state == state_label]
        if not indices:
            continue
        plt.scatter(
            scores[indices, 0],
            scores[indices, 1],
            s=55,
            alpha=0.8,
            label=state_label,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )

    plt.axhline(0.0, color="0.7", linewidth=0.8)
    plt.axvline(0.0, color="0.7", linewidth=0.8)
    plt.xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% variance)")
    plt.title("Pre-ML PCA View of 12 AIR States from RF Features")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def plot_top_feature_boxplots(
    records: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """
    Save boxplots for the features that separate the most state pairs.

    Boxplots are intentionally simple: if boxes for states barely overlap,
    the feature is useful for engineering separability.
    """
    top_features = [str(row["feature"]) for row in feature_rows[:4]]
    states = [
        state_id(interference_code, mode_code)
        for interference_code in INTERFERENCE_ORDER
        for mode_code in MODE_ORDER
    ]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for axis, feature_name in zip(axes, top_features):
        grouped_values = [
            [
                float(record[feature_name])
                for record in records
                if str(record["state"]) == state_label
            ]
            for state_label in states
        ]
        axis.boxplot(grouped_values, tick_labels=states, showfliers=True)
        axis.set_title(feature_name)
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle("Top RF Features for State Separability", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_variation_rankings(
    mode_variation_rows: list[dict[str, object]],
    interference_variation_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save bar charts ranking which features vary most by mode/interference."""
    top_count = min(8, len(FEATURE_NAMES))
    mode_top = mode_variation_rows[:top_count]
    interference_top = interference_variation_rows[:top_count]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(
        [str(row["feature"]) for row in reversed(mode_top)],
        [float(row["normalized_variation_score"]) for row in reversed(mode_top)],
        color="tab:blue",
    )
    axes[0].set_title("Feature Variation Across Flight Modes")
    axes[0].set_xlabel("Normalized variation score")

    axes[1].barh(
        [str(row["feature"]) for row in reversed(interference_top)],
        [float(row["normalized_variation_score"]) for row in reversed(interference_top)],
        color="tab:orange",
    )
    axes[1].set_title("Feature Variation Across Interference Conditions")
    axes[1].set_xlabel("Normalized variation score")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(figure)


def write_engineering_report(
    output_dir: Path,
    records: list[dict[str, object]],
    engineering_summary: dict[str, object],
    central_engineering_summary: dict[str, object],
    feature_pair_rows: list[dict[str, object]],
    central_feature_pair_rows: list[dict[str, object]],
    mode_variation_rows: list[dict[str, object]],
    interference_variation_rows: list[dict[str, object]],
) -> None:
    """
    Write a plain-English engineering interpretation report.

    The report avoids abstract ML-style scores as the main conclusion. It
    focuses on direct evidence: visible plots and non-overlapping feature
    ranges between operating states.
    """
    total_pairs = int(engineering_summary["total_state_pairs"])
    strong_pairs = int(engineering_summary["strong_pairs"])
    partial_pairs = int(engineering_summary["partial_pairs"])
    overlap_pairs = int(engineering_summary["overlap_pairs"])
    evidence_percent = float(engineering_summary["strong_or_partial_percent"])
    central_strong_pairs = int(central_engineering_summary["strong_pairs"])
    central_partial_pairs = int(central_engineering_summary["partial_pairs"])
    central_overlap_pairs = int(central_engineering_summary["overlap_pairs"])
    central_evidence_percent = float(
        central_engineering_summary["strong_or_partial_percent"]
    )
    total_feature_rows = len(records)
    source_recording_count = len({str(record["file_path"]) for record in records})
    segment_counts_by_state = {
        state_label: sum(1 for record in records if str(record["state"]) == state_label)
        for state_label in {
            str(record["state"])
            for record in records
        }
    }
    min_segments_per_state = min(segment_counts_by_state.values())
    max_segments_per_state = max(segment_counts_by_state.values())

    top_pair_features = "\n".join(
        [
            (
                f"- `{row['feature']}` separates {row['state_pairs_separated']} "
                f"of {row['total_state_pairs']} state pairs "
                f"({float(row['pair_separation_percent']):.1f}%)"
            )
            for row in feature_pair_rows[:5]
        ]
    )
    central_top_pair_features = "\n".join(
        [
            (
                f"- `{row['feature']}` separates {row['state_pairs_separated']} "
                f"of {row['total_state_pairs']} state pairs "
                f"({float(row['pair_separation_percent']):.1f}%)"
            )
            for row in central_feature_pair_rows[:5]
        ]
    )
    top_mode_features = "\n".join(
        [
            f"- `{row['feature']}`"
            for row in mode_variation_rows[:5]
        ]
    )
    top_interference_features = "\n".join(
        [
            f"- `{row['feature']}`"
            for row in interference_variation_rows[:5]
        ]
    )

    if central_evidence_percent >= 80.0:
        conclusion = (
            "The AIR states show strong pre-ML separability evidence from the "
            "current RF feature set."
        )
    elif central_evidence_percent >= 50.0:
        conclusion = (
            "The AIR states show partial pre-ML separability evidence. Some "
            "states are clearly distinguishable, but some still overlap."
        )
    else:
        conclusion = (
            "The current RF feature set shows limited direct separability. "
            "More segment sampling and richer features are needed before "
            "claiming all 12 states are separable."
        )

    report = f"""# Stage 4 Engineering Separability Report

## Research Question

Do AIR drone RF recordings show distinguishable characteristics across the 12
operating states?

The 12 states are:

- 4 interference conditions: `00`, `01`, `10`, `11`
- 3 flight modes: `ON`, `HO`, `FY`

## Engineer-Readable Separability Criterion

For every pair of states, the script checks whether any RF feature has
non-overlapping observed ranges across the extracted segments for each state.

Labels:

- `STRONG`: 3 or more RF features separate the pair
- `PARTIAL`: 1 or 2 RF features separate the pair
- `OVERLAP`: no current feature fully separates the pair

This is not model training. It is direct feature inspection.

## Result

- Source recordings processed: {source_recording_count}
- Feature rows extracted: {total_feature_rows}
- Segments per state: {min_segments_per_state} to {max_segments_per_state}
- Total state pairs checked: {total_pairs}
- Strong pairs: {strong_pairs}
- Partial pairs: {partial_pairs}
- Overlapping pairs: {overlap_pairs}
- State pairs with at least some direct feature evidence: {evidence_percent:.1f}%

## Result Using Central 80% Ranges

The central 80% range uses the 10th to 90th percentile for each feature. This
is less affected by unusual individual segments.

- Strong pairs: {central_strong_pairs}
- Partial pairs: {central_partial_pairs}
- Overlapping pairs: {central_overlap_pairs}
- State pairs with at least some typical-feature evidence: {central_evidence_percent:.1f}%

## Best Features for Direct State Separation

{top_pair_features}

## Best Features for Typical Central-Range Separation

{central_top_pair_features}

## Features That Vary Most Across Flight Modes

{top_mode_features}

## Features That Vary Most Across Interference Conditions

{top_interference_features}

## Engineering Conclusion

{conclusion}

The most useful plots to inspect are:

- `pca_12_state_separability.png`
- `top_feature_boxplots_by_state.png`
- `state_feature_heatmap.png`

The most useful tables to inspect are:

- `pairwise_engineering_separability.csv`
- `central80_pairwise_engineering_separability.csv`
- `feature_pair_separation_rank.csv`
- `central80_feature_pair_separation_rank.csv`
- `feature_intervals_by_state.csv`
"""
    (output_dir / "engineering_separability_report.md").write_text(
        report,
        encoding="utf-8",
    )


def process_air_recordings(args: argparse.Namespace) -> list[dict[str, object]]:
    """Load AIR files and extract one feature row per 20 ms segment."""
    air_files = find_air_dat_files(args.data_root)
    if args.max_files is not None:
        air_files = air_files[: args.max_files]

    if not air_files:
        raise FileNotFoundError(f"No AIR files found under {args.data_root}")

    segment_samples = int(round(args.segment_ms / 1000.0 * args.sample_rate))
    start_sample = int(round(args.start_ms / 1000.0 * args.sample_rate))

    print("\n--- Stage 4 AIR state analysis ---")
    print(f"AIR files to process: {len(air_files)}")
    print(f"First possible segment start: {args.start_ms:.3f} ms")
    print(f"Segment duration: {args.segment_ms:.3f} ms")
    print(f"Requested segments per file: {args.segments_per_file}")
    print(f"Samples per segment: {segment_samples}")

    records = []
    for file_index, file_path in enumerate(air_files, start=1):
        metadata = parse_recording_metadata(file_path)
        interference_code = str(metadata["interference_code"])
        mode_code = str(metadata["mode_code"])

        total_raw_floats = count_raw_floats_from_file_size(file_path)
        total_iq_samples = total_raw_floats // 2
        segment_starts = choose_evenly_spaced_segment_starts(
            total_iq_samples,
            segment_samples,
            start_sample,
            args.segments_per_file,
        )

        file_records = []
        for segment_index, segment_start_sample in enumerate(segment_starts):
            available_samples = total_iq_samples - segment_start_sample
            samples_to_read = min(segment_samples, available_samples)
            iq_signal = load_iq_segment(file_path, segment_start_sample, samples_to_read)

            features = extract_features(
                iq_signal,
                args.sample_rate,
                args.psd_nperseg,
                args.psd_overlap,
                args.spectrogram_nperseg,
                args.spectrogram_overlap,
                args.dc_exclusion_hz,
            )

            segment_start_ms = segment_start_sample / args.sample_rate * 1e3
            record: dict[str, object] = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "model": metadata["model"],
                "interference_code": interference_code,
                "interference_label": INTERFERENCE_LABELS[interference_code],
                "mode_code": mode_code,
                "mode_label": FLIGHT_MODE_LABELS[mode_code],
                "state": state_id(interference_code, mode_code),
                "recording_index": metadata["file_index"],
                "segment_index": segment_index,
                "segments_per_file": len(segment_starts),
                "segment_start_sample": segment_start_sample,
                "segment_start_ms": segment_start_ms,
                "segment_duration_ms": args.segment_ms,
                "samples_read": iq_signal.size,
            }
            record.update(features)
            file_records.append(record)
            records.append(record)

        print(
            f"[{file_index:02d}/{len(air_files):02d}] "
            f"{state_id(interference_code, mode_code)} {file_path.name} "
            f"segments={len(file_records)} "
            f"peak_mean={np.mean([float(record['peak_frequency_mhz']) for record in file_records]):.2f} MHz "
            f"entropy_mean={np.mean([float(record['spectral_entropy']) for record in file_records]):.3f}"
        )

    return records


def write_outputs(records: list[dict[str, object]], output_dir: Path) -> None:
    """Create all CSV tables and summary figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state_summary = make_state_summary(records)
    analysis_a_rows = make_analysis_a_rows(state_summary)
    analysis_b_rows = make_analysis_b_rows(state_summary)
    mode_variation_rows = rank_feature_variation(records, "mode_code")
    interference_variation_rows = rank_feature_variation(records, "interference_code")
    separability_rows = estimate_state_separability(records)
    interval_rows = make_feature_interval_rows(records)
    pair_rows, feature_pair_rows, engineering_summary = make_pairwise_engineering_separability(
        records
    )
    (
        central_pair_rows,
        central_feature_pair_rows,
        central_engineering_summary,
    ) = make_pairwise_central_interval_separability(records)

    metadata_fields = [
        "file_path",
        "file_name",
        "model",
        "interference_code",
        "interference_label",
        "mode_code",
        "mode_label",
        "state",
        "recording_index",
        "segment_index",
        "segments_per_file",
        "segment_start_sample",
        "segment_start_ms",
        "segment_duration_ms",
        "samples_read",
    ]
    write_csv(
        output_dir / "per_record_features.csv",
        records,
        metadata_fields + FEATURE_NAMES,
    )

    state_summary_fields = [
        "state",
        "interference_code",
        "interference_label",
        "mode_code",
        "mode_label",
        "source_recording_count",
        "segment_count",
    ]
    for feature_name in FEATURE_NAMES:
        state_summary_fields.extend([f"{feature_name}_mean", f"{feature_name}_std"])
    write_csv(output_dir / "state_summary_features.csv", state_summary, state_summary_fields)

    write_csv(
        output_dir / "analysis_A_fix_mode_compare_interference.csv",
        analysis_a_rows,
        [
            "analysis",
            "mode_code",
            "mode_label",
            "feature",
            "00_mean",
            "01_mean",
            "10_mean",
            "11_mean",
            "range",
            "max_condition",
            "min_condition",
        ],
    )

    write_csv(
        output_dir / "analysis_B_fix_interference_compare_mode.csv",
        analysis_b_rows,
        [
            "analysis",
            "interference_code",
            "interference_label",
            "feature",
            "ON_mean",
            "HO_mean",
            "FY_mean",
            "range",
            "max_mode",
            "min_mode",
        ],
    )

    write_csv(
        output_dir / "feature_variation_by_mode.csv",
        mode_variation_rows,
        [
            "group_key",
            "feature",
            "group_mean_range",
            "overall_std",
            "normalized_variation_score",
        ],
    )
    write_csv(
        output_dir / "feature_variation_by_interference.csv",
        interference_variation_rows,
        [
            "group_key",
            "feature",
            "group_mean_range",
            "overall_std",
            "normalized_variation_score",
        ],
    )
    write_csv(
        output_dir / "state_separability_pre_ml.csv",
        separability_rows,
        [
            "state",
            "nearest_state",
            "nearest_centroid_distance",
            "within_state_radius",
            "separation_ratio",
            "pre_ml_separability",
        ],
    )
    write_csv(
        output_dir / "feature_intervals_by_state.csv",
        interval_rows,
        [
            "state",
            "interference_code",
            "mode_code",
            "feature",
            "min",
            "mean",
            "max",
            "std",
        ],
    )
    write_csv(
        output_dir / "pairwise_engineering_separability.csv",
        pair_rows,
        [
            "state_a",
            "state_b",
            "non_overlapping_feature_count",
            "separating_features",
            "engineering_evidence",
        ],
    )
    write_csv(
        output_dir / "feature_pair_separation_rank.csv",
        feature_pair_rows,
        [
            "feature",
            "state_pairs_separated",
            "total_state_pairs",
            "pair_separation_percent",
        ],
    )
    write_csv(
        output_dir / "central80_pairwise_engineering_separability.csv",
        central_pair_rows,
        [
            "state_a",
            "state_b",
            "central_interval",
            "non_overlapping_feature_count",
            "separating_features",
            "engineering_evidence",
        ],
    )
    write_csv(
        output_dir / "central80_feature_pair_separation_rank.csv",
        central_feature_pair_rows,
        [
            "feature",
            "state_pairs_separated",
            "total_state_pairs",
            "pair_separation_percent",
        ],
    )

    plot_state_feature_heatmap(
        state_summary,
        output_dir / "state_feature_heatmap.png",
    )
    plot_pca_separability(
        records,
        output_dir / "pca_12_state_separability.png",
    )
    plot_top_feature_boxplots(
        records,
        feature_pair_rows,
        output_dir / "top_feature_boxplots_by_state.png",
    )
    plot_variation_rankings(
        mode_variation_rows,
        interference_variation_rows,
        output_dir / "feature_variation_rankings.png",
    )
    write_engineering_report(
        output_dir,
        records,
        engineering_summary,
        central_engineering_summary,
        feature_pair_rows,
        central_feature_pair_rows,
        mode_variation_rows,
        interference_variation_rows,
    )

    print("\n--- Most mode-sensitive features ---")
    for row in mode_variation_rows[:5]:
        print(
            f"{row['feature']}: "
            f"score={float(row['normalized_variation_score']):.3f}"
        )

    print("\n--- Most interference-sensitive features ---")
    for row in interference_variation_rows[:5]:
        print(
            f"{row['feature']}: "
            f"score={float(row['normalized_variation_score']):.3f}"
        )

    ratios = [
        float(row["separation_ratio"])
        for row in separability_rows
        if row["separation_ratio"] != ""
    ]
    print("\n--- Pre-ML separability summary ---")
    if ratios:
        print(f"Median state separation ratio: {float(np.median(ratios)):.3f}")
        print(f"Minimum state separation ratio: {float(np.min(ratios)):.3f}")
        print(
            "Interpretation: ratios above 1 suggest state centroids are farther "
            "apart than within-state spread; ratios above 2 are stronger evidence."
        )
    else:
        print("Not enough states were processed to estimate separability.")

    print("\n--- Engineering separability evidence ---")
    print(f"Total state pairs checked: {engineering_summary['total_state_pairs']}")
    print(f"Strong pairs, 3+ non-overlapping features: {engineering_summary['strong_pairs']}")
    print(f"Partial pairs, 1-2 non-overlapping features: {engineering_summary['partial_pairs']}")
    print(f"Overlapping pairs, no fully separating feature: {engineering_summary['overlap_pairs']}")
    print(
        "Strong or partial evidence: "
        f"{engineering_summary['strong_or_partial_percent']:.1f}% of state pairs"
    )
    print("\n--- Central 80% separability evidence ---")
    print(f"Total state pairs checked: {central_engineering_summary['total_state_pairs']}")
    print(f"Strong pairs, 3+ non-overlapping features: {central_engineering_summary['strong_pairs']}")
    print(f"Partial pairs, 1-2 non-overlapping features: {central_engineering_summary['partial_pairs']}")
    print(f"Overlapping pairs: {central_engineering_summary['overlap_pairs']}")
    print(
        "Strong or partial typical-feature evidence: "
        f"{central_engineering_summary['strong_or_partial_percent']:.1f}% of state pairs"
    )

    print(f"\nSaved outputs to: {output_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    """Parse stage-4 analysis settings."""
    parser = argparse.ArgumentParser(
        description="Batch AIR RF feature analysis across 12 operating states."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root folder containing DroneDetect AIR data. Default: data, or the "
            "DRONEDETECT_DATASET_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where CSV tables and plots will be saved.",
    )
    parser.add_argument(
        "--segment-ms",
        type=float,
        default=DEFAULT_SEGMENT_MS,
        help="IQ segment duration in milliseconds. Default: 20",
    )
    parser.add_argument(
        "--start-ms",
        type=float,
        default=DEFAULT_START_MS,
        help=(
            "Earliest allowed segment start within each recording in milliseconds. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=DEFAULT_SEGMENTS_PER_FILE,
        help=(
            "Number of evenly spaced segments to extract from each recording. "
            "Default: 10"
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="IQ sample rate in Hz. Default: 60e6",
    )
    parser.add_argument(
        "--center-frequency",
        type=float,
        default=DEFAULT_CENTER_FREQUENCY_HZ,
        help="RF centre frequency in Hz. Used for interpretation. Default: 2.4375e9",
    )
    parser.add_argument(
        "--dc-exclusion-hz",
        type=float,
        default=DEFAULT_DC_EXCLUSION_HZ,
        help="Frequency span around 0 Hz excluded from selected spectral features.",
    )
    parser.add_argument(
        "--psd-nperseg",
        type=int,
        default=DEFAULT_PSD_NPERSEG,
        help="Welch PSD window length. Default: 8192",
    )
    parser.add_argument(
        "--psd-overlap",
        type=int,
        default=DEFAULT_PSD_OVERLAP,
        help="Welch PSD overlap. Default: 4096",
    )
    parser.add_argument(
        "--spectrogram-nperseg",
        type=int,
        default=DEFAULT_SPECTROGRAM_NPERSEG,
        help="Spectrogram window length. Default: 4096",
    )
    parser.add_argument(
        "--spectrogram-overlap",
        type=int,
        default=DEFAULT_SPECTROGRAM_OVERLAP,
        help="Spectrogram overlap. Default: 2048",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick debugging. Default: process all AIR files.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Fail early on invalid analysis settings."""
    if args.segment_ms <= 0:
        raise ValueError("--segment-ms must be greater than zero.")
    if args.start_ms < 0:
        raise ValueError("--start-ms cannot be negative.")
    if args.segments_per_file <= 0:
        raise ValueError("--segments-per-file must be greater than zero.")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero.")
    if args.dc_exclusion_hz < 0:
        raise ValueError("--dc-exclusion-hz cannot be negative.")
    if args.psd_nperseg < 4 or args.spectrogram_nperseg < 4:
        raise ValueError("PSD and spectrogram window lengths must be at least 4.")
    if args.psd_overlap < 0 or args.spectrogram_overlap < 0:
        raise ValueError("PSD and spectrogram overlaps cannot be negative.")
    if args.psd_overlap >= args.psd_nperseg:
        raise ValueError("--psd-overlap must be smaller than --psd-nperseg.")
    if args.spectrogram_overlap >= args.spectrogram_nperseg:
        raise ValueError(
            "--spectrogram-overlap must be smaller than --spectrogram-nperseg."
        )
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be greater than zero when supplied.")


def main() -> None:
    """Run the stage-4 AIR state separability analysis."""
    args = parse_arguments()
    validate_arguments(args)
    records = process_air_recordings(args)
    write_outputs(records, args.output_dir)


if __name__ == "__main__":
    main()
