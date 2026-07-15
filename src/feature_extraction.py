"""
Extract interpretable RF features for AIR UAV behaviour recognition.

This module bridges signal processing and machine learning. It takes complex
IQ segments, computes FFT/PSD/spectrogram representations, and summarizes them
as compact RF features. The same functions are used during training and
prediction, preventing feature drift between validation and deployment.

Notes:
    The feature set intentionally remains engineering-readable:

    * RMS power and signal energy describe received signal strength.
    * Peak frequency and spectral centroid describe where RF energy lives.
    * Occupied bandwidth describes how wide the active spectrum is.
    * Spectral entropy describes concentrated versus spread energy.
    * Spectrogram activity and temporal variability describe burstiness and
      time-frequency structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from src.iq_loader import (
    DEFAULT_CENTER_FREQUENCY_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    RecordingMetadata,
    choose_segment_starts,
    count_iq_samples,
    find_air_dat_files,
    load_iq_segment,
    parse_recording_metadata,
)
from src.signal_processing import (
    DEFAULT_FFT_SAMPLES,
    compute_fft_power,
    compute_psd,
    compute_spectrogram_power,
    occupied_bandwidth_hz,
    remove_dc,
    spectral_entropy,
    valid_frequency_mask,
)


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


@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration shared by training and prediction feature extraction.

    Attributes:
        sample_rate_hz: IQ sample rate in Hz.
        center_frequency_hz: SDR centre frequency in Hz.
        segment_ms: Duration of one analysis segment.
        start_ms: Earliest segment start time.
        segments_per_recording: Number of segments per recording. Values less
            than or equal to zero mean all non-overlapping segments.
        fft_samples: Maximum FFT length for peak-frequency estimation.
        psd_nperseg: Welch PSD window length.
        psd_overlap: Welch PSD overlap.
        spectrogram_nperseg: Spectrogram short-time FFT length.
        spectrogram_overlap: Spectrogram overlap.
        dc_exclusion_hz: Centre-frequency exclusion band used for spectral
            content features.

    Notes:
        This object is serialized into `models/feature_config.json`, so a
        saved model carries the exact feature settings used during training.
    """

    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ
    segment_ms: float = 20.0
    start_ms: float = 0.0
    segments_per_recording: int = 10
    fft_samples: int = DEFAULT_FFT_SAMPLES
    psd_nperseg: int = 8192
    psd_overlap: int = 4096
    spectrogram_nperseg: int = 4096
    spectrogram_overlap: int = 2048
    dc_exclusion_hz: float = 250_000.0

    @property
    def segment_samples(self) -> int:
        """
        Number of complex IQ samples in one segment.

        Returns:
            Segment duration converted from milliseconds to samples.
        """
        return int(round(self.segment_ms / 1000.0 * self.sample_rate_hz))

    @property
    def first_start_sample(self) -> int:
        """
        First allowed segment start, expressed in samples.

        Returns:
            Start time converted from milliseconds to samples.
        """
        return int(round(self.start_ms / 1000.0 * self.sample_rate_hz))

    def with_segments(self, segments_per_recording: int) -> "FeatureConfig":
        """
        Return a copy with a different segment count.

        Args:
            segments_per_recording: Replacement segment count.

        Returns:
            New `FeatureConfig` with all other fields unchanged.
        """
        return replace(self, segments_per_recording=segments_per_recording)

    def to_dict(self) -> dict[str, float | int]:
        """
        Serialize config for `feature_config.json`.

        Returns:
            Dictionary containing dataclass fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "FeatureConfig":
        """
        Create a config from saved model metadata.

        Args:
            values: Dictionary loaded from `feature_config.json`.

        Returns:
            `FeatureConfig` built from matching keys.
        """
        field_names = set(cls.__dataclass_fields__)
        filtered = {key: values[key] for key in values if key in field_names}
        return cls(**filtered)


def _safe_peak_index(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    dc_exclusion_hz: float,
) -> int:
    """
    Find the strongest non-centre spectral bin.

    Args:
        frequency_hz: Frequency-bin offsets.
        power: Power values aligned with `frequency_hz`.
        dc_exclusion_hz: Centre exclusion half-width.

    Returns:
        Index of the strongest valid bin. If every bin is excluded, the
        strongest bin over the full spectrum is returned.
    """
    mask = valid_frequency_mask(frequency_hz, dc_exclusion_hz)
    if not np.any(mask):
        return int(np.argmax(power))
    valid_indices = np.where(mask)[0]
    return int(valid_indices[int(np.argmax(power[mask]))])


def extract_segment_features(
    iq_signal: np.ndarray,
    config: FeatureConfig,
) -> dict[str, float]:
    """
    Extract RF features from one 20 ms complex IQ segment.

    Args:
        iq_signal: Complex IQ samples for one segment.
        config: Feature extraction settings.

    Returns:
        Dictionary containing the fixed feature set listed in `FEATURE_NAMES`.

    Notes:
        rms_power and signal_energy describe signal strength.
        peak and centroid frequencies describe where the RF energy lives.
        occupied bandwidth describes how wide the active signal is.
        entropy describes concentrated versus spread spectral energy.
        spectrogram features describe time-varying activity and bursts.
    """
    iq_without_dc = remove_dc(iq_signal)
    time_power = np.abs(iq_without_dc) ** 2

    fft_frequency_hz, fft_power = compute_fft_power(
        iq_signal,
        config.sample_rate_hz,
        config.fft_samples,
    )
    psd_frequency_hz, psd_power = compute_psd(
        iq_signal,
        config.sample_rate_hz,
        config.psd_nperseg,
        config.psd_overlap,
    )
    _, _, spectrogram_power = compute_spectrogram_power(
        iq_signal,
        config.sample_rate_hz,
        config.spectrogram_nperseg,
        config.spectrogram_overlap,
    )

    fft_peak_index = _safe_peak_index(
        fft_frequency_hz,
        fft_power,
        config.dc_exclusion_hz,
    )
    psd_peak_index = _safe_peak_index(
        psd_frequency_hz,
        psd_power,
        config.dc_exclusion_hz,
    )

    psd_mask = valid_frequency_mask(psd_frequency_hz, config.dc_exclusion_hz)
    if not np.any(psd_mask):
        psd_mask = np.ones(psd_frequency_hz.shape, dtype=bool)

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


def metadata_to_row(metadata: RecordingMetadata) -> dict[str, object]:
    """
    Convert decoded metadata to stable row fields.

    Args:
        metadata: Decoded recording metadata.

    Returns:
        Dictionary suitable for CSV output or feature-row metadata.
    """
    return {
        "file_path": str(metadata.file_path),
        "file_name": metadata.file_name,
        "model": metadata.model,
        "interference_code": metadata.interference_code,
        "interference_label": metadata.interference_label,
        "mode_code": metadata.mode_code,
        "mode_label": metadata.mode_label,
        "state": metadata.state,
        "recording_index": metadata.recording_index,
    }


def unknown_metadata_row(file_path: Path) -> dict[str, object]:
    """
    Metadata placeholder for unlabeled deployment recordings.

    Args:
        file_path: Input `.dat` path.

    Returns:
        Metadata dictionary with unknown label fields left empty.

    Notes:
        Training and validation need true labels, but production prediction should
        still work when the incoming .dat filename does not encode ON/HO/FY.
    """
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "model": file_path.stem.split("_", maxsplit=1)[0].upper(),
        "interference_code": "",
        "interference_label": "",
        "mode_code": "",
        "mode_label": "",
        "state": "",
        "recording_index": "",
    }


def extract_recording_feature_rows(
    file_path: Path,
    config: FeatureConfig,
    require_metadata: bool = True,
) -> list[dict[str, object]]:
    """
    Extract RF feature rows from selected segments of one recording.

    Args:
        file_path: Path to one AIR `.dat` recording.
        config: Feature extraction settings.
        require_metadata: If True, labels must be decoded from the path. If
            False, unknown label fields are allowed for deployment prediction.

    Returns:
        One dictionary per analysed segment, containing metadata and RF features.

    Notes:
        The same function is used by both training and prediction, which prevents
        feature-definition drift between validation and deployment.
    """
    try:
        base_row = metadata_to_row(parse_recording_metadata(file_path))
    except ValueError:
        if require_metadata:
            raise
        base_row = unknown_metadata_row(file_path)

    total_iq_samples = count_iq_samples(file_path)
    segment_starts = choose_segment_starts(
        total_iq_samples=total_iq_samples,
        segment_samples=config.segment_samples,
        first_start_sample=config.first_start_sample,
        segments_per_recording=config.segments_per_recording,
    )

    rows: list[dict[str, object]] = []
    for segment_index, segment_start_sample in enumerate(segment_starts):
        samples_to_read = min(config.segment_samples, total_iq_samples - segment_start_sample)
        iq_signal = load_iq_segment(file_path, segment_start_sample, samples_to_read)
        features = extract_segment_features(iq_signal, config)

        row = dict(base_row)
        row.update(
            {
                "segment_index": segment_index,
                "segments_per_recording": len(segment_starts),
                "segment_start_sample": segment_start_sample,
                "segment_start_ms": segment_start_sample / config.sample_rate_hz * 1e3,
                "segment_duration_ms": config.segment_ms,
                "samples_read": iq_signal.size,
            }
        )
        row.update(features)
        rows.append(row)

    return rows


def build_air_feature_rows(
    dataset_root: Path,
    config: FeatureConfig,
    max_files: int | None = None,
    progress_callback: Callable[[int, int, Path, list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    """
    Extract feature rows from every AIR recording in a dataset folder.

    Args:
        dataset_root: Root folder containing AIR `.dat` recordings.
        config: Feature extraction settings.
        max_files: Optional cap used for quick smoke tests.
        progress_callback: Optional function called after each file.

    Returns:
        Segment-level feature rows from all selected recordings.
    """
    air_files = find_air_dat_files(dataset_root)
    if max_files is not None:
        air_files = air_files[:max_files]
    if not air_files:
        raise FileNotFoundError(f"No AIR .dat files found under {dataset_root}")

    all_rows: list[dict[str, object]] = []
    for file_index, file_path in enumerate(air_files, start=1):
        file_rows = extract_recording_feature_rows(
            file_path,
            config,
            require_metadata=True,
        )
        all_rows.extend(file_rows)
        if progress_callback is not None:
            progress_callback(file_index, len(air_files), file_path, file_rows)
    return all_rows


def feature_rows_to_matrix(
    rows: list[dict[str, object]],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    Convert feature dictionaries to a numeric matrix.

    Args:
        rows: Feature dictionaries.
        feature_names: Ordered feature names. Defaults to `FEATURE_NAMES`.

    Returns:
        Two-dimensional numeric matrix with shape `(rows, features)`.

    Raises:
        ValueError: If any feature value is NaN or infinite.

    Notes:
        Fixed ordering is critical because the scaler and classifier expect the
        same feature order used during training.
    """
    feature_names = feature_names or FEATURE_NAMES
    matrix = np.array(
        [[float(row[feature_name]) for feature_name in feature_names] for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return matrix
