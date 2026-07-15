"""
Load DroneDetect AIR IQ recordings and decode dataset metadata.

This module is the entry point from raw binary RF recordings into the
engineering pipeline. It knows how to count samples, read small windows from
large `.dat` files, reconstruct complex IQ samples, and decode labels from
DroneDetect-style filenames/folders when labels are available.

Notes:
    DroneDetect `.dat` files store interleaved float32 baseband samples:

        I0, Q0, I1, Q1, I2, Q2, ...

    Two adjacent float32 values form one complex sample `I + jQ`. The I
    component is in phase, and Q is the quadrature component shifted by
    90 degrees. Reconstructing complex IQ is required before FFT, PSD,
    spectrogram, and RF feature extraction can be meaningful.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATASET_ROOT = Path(os.environ.get("DRONEDETECT_DATASET_ROOT", "data"))
DEFAULT_SAMPLE_RATE_HZ = 60_000_000.0
DEFAULT_CENTER_FREQUENCY_HZ = 2_437_500_000.0
FLOAT32_BYTES = np.dtype(np.float32).itemsize

INTERFERENCE_LABELS = {
    "00": "Clean",
    "01": "Bluetooth",
    "10": "WiFi",
    "11": "Bluetooth + WiFi",
}

INTERFERENCE_FOLDER_TO_CODE = {
    "CLEAN": "00",
    "BLUE": "01",
    "BLUETOOTH": "01",
    "WIFI": "10",
    "WI-FI": "10",
    "BOTH": "11",
}

MODE_FIELD_TO_CODE = {
    "00": "ON",
    "01": "HO",
    "10": "FY",
    "ON": "ON",
    "HO": "HO",
    "FY": "FY",
}

FLIGHT_MODE_LABELS = {
    "ON": "Switched on",
    "HO": "Hovering",
    "FY": "Flying",
}

INTERFERENCE_ORDER = {"00": 0, "01": 1, "10": 2, "11": 3}
MODE_ORDER = {"ON": 0, "HO": 1, "FY": 2}


@dataclass(frozen=True)
class RecordingMetadata:
    """
    Metadata decoded from a DroneDetect AIR path.

    Attributes:
        file_path: Full path to the source `.dat` recording.
        file_name: Name of the recording file.
        model: Drone/model short code, expected to be `AIR` for this project.
        interference_code: DroneDetect interference code: `00`, `01`, `10`, or `11`.
        interference_label: Human-readable interference label.
        mode_code: Flight-mode code: `ON`, `HO`, or `FY`.
        mode_label: Human-readable flight-mode label.
        state: Combined 12-state label, for example `00_ON`.
        recording_index: Original recording index used for grouped validation.

    Notes:
        Metadata is used for training labels and validation grouping only. The
        prediction path can process unlabeled files without requiring this
        metadata.
    """

    file_path: Path
    file_name: str
    model: str
    interference_code: str
    interference_label: str
    mode_code: str
    mode_label: str
    state: str
    recording_index: str


def count_raw_floats_from_file_size(file_path: Path) -> int:
    """
    Count raw float32 values without loading the full recording.

    Args:
        file_path: Path to a DroneDetect `.dat` file.

    Returns:
        Number of float32 values stored in the file.

    Raises:
        ValueError: If the file size is not divisible by the float32 byte size.

    Notes:
        One complex IQ sample is stored as two float32 values. This count lets
        us determine recording length and segment positions safely.
    """
    file_size_bytes = file_path.stat().st_size
    if file_size_bytes % FLOAT32_BYTES != 0:
        raise ValueError(
            f"{file_path} size is not divisible by 4 bytes; expected float32 IQ data."
        )
    return file_size_bytes // FLOAT32_BYTES


def count_iq_samples(file_path: Path) -> int:
    """
    Return the number of complex IQ samples in a `.dat` file.

    Args:
        file_path: Path to an interleaved float32 IQ recording.

    Returns:
        Number of complete complex IQ samples.

    Raises:
        ValueError: If the raw float count is odd and cannot form I/Q pairs.
    """
    raw_float_count = count_raw_floats_from_file_size(file_path)
    if raw_float_count % 2 != 0:
        raise ValueError(f"{file_path} has an odd number of floats; IQ pairs are broken.")
    return raw_float_count // 2


def parse_recording_metadata(file_path: Path) -> RecordingMetadata:
    """
    Decode AIR model, interference condition, flight mode, and recording index.

    Args:
        file_path: Path to a DroneDetect AIR `.dat` file.

    Returns:
        Decoded `RecordingMetadata`.

    Raises:
        ValueError: If interference, flight mode, or recording index cannot be
            decoded from the path/name.

    Notes:
        This function uses only labels in the path and filename. It does not
        inspect the RF signal, so it is safe for training labels and validation
        grouping. Production prediction uses `require_metadata=False` in the
        feature extractor so unknown files can still be classified.
    """
    upper_name = file_path.name.upper()
    upper_parts = [part.upper() for part in file_path.parts]
    filename_fields = upper_name.replace(".DAT", "").split("_")

    model = filename_fields[0] if filename_fields else "UNKNOWN"
    interference_code = ""
    mode_code = ""
    recording_index = ""

    if len(filename_fields) >= 2 and len(filename_fields[1]) >= 4:
        interference_code = filename_fields[1][:2]
        mode_code = MODE_FIELD_TO_CODE.get(filename_fields[1][2:4], filename_fields[1][2:4])

    if len(filename_fields) >= 3:
        recording_index = filename_fields[2]

    for folder_name, folder_interference_code in INTERFERENCE_FOLDER_TO_CODE.items():
        if folder_name in upper_parts:
            interference_code = folder_interference_code
            break

    # Some dataset copies use folders such as AIR_ON, AIR_HO, and AIR_FY.
    for folder_mode_code in ("ON", "HO", "FY"):
        if any(
            part == folder_mode_code or part.endswith(f"_{folder_mode_code}")
            for part in upper_parts
        ):
            mode_code = folder_mode_code
            break

    if interference_code not in INTERFERENCE_LABELS:
        raise ValueError(f"Could not decode interference condition from {file_path}")
    if mode_code not in FLIGHT_MODE_LABELS:
        raise ValueError(f"Could not decode flight mode from {file_path}")
    if not recording_index:
        raise ValueError(f"Could not decode recording index from {file_path}")

    state = f"{interference_code}_{mode_code}"
    return RecordingMetadata(
        file_path=file_path,
        file_name=file_path.name,
        model=model,
        interference_code=interference_code,
        interference_label=INTERFERENCE_LABELS[interference_code],
        mode_code=mode_code,
        mode_label=FLIGHT_MODE_LABELS[mode_code],
        state=state,
        recording_index=recording_index,
    )


def sort_air_files_for_analysis(file_paths: list[Path]) -> list[Path]:
    """
    Sort AIR recordings into a stable engineering-analysis order.

    Args:
        file_paths: Candidate AIR `.dat` paths.

    Returns:
        Paths sorted by interference condition, flight mode, recording index,
        and path string.

    Notes:
        Stable ordering makes generated CSVs and reports reproducible across
        repeated runs on the same dataset.
    """

    def sort_key(file_path: Path) -> tuple[int, int, str, str]:
        """Return the sortable metadata tuple for one AIR file."""
        metadata = parse_recording_metadata(file_path)
        return (
            INTERFERENCE_ORDER[metadata.interference_code],
            MODE_ORDER[metadata.mode_code],
            metadata.recording_index,
            str(file_path),
        )

    return sorted(file_paths, key=sort_key)


def find_air_dat_files(dataset_root: Path) -> list[Path]:
    """
    Find DroneDetect AIR `.dat` recordings under a dataset root.

    Args:
        dataset_root: Folder containing the DroneDetect dataset or AIR subset.

    Returns:
        Sorted list of AIR `.dat` files.

    Raises:
        FileNotFoundError: If `dataset_root` does not exist.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}. "
            "Pass --data-root or set DRONEDETECT_DATASET_ROOT."
        )
    return sort_air_files_for_analysis(list(dataset_root.rglob("AIR*.dat")))


def load_iq_segment(file_path: Path, start_sample: int, sample_count: int) -> np.ndarray:
    """
    Load one complex IQ segment from an interleaved float32 DroneDetect file.

    Args:
        file_path: Path to the `.dat` IQ recording.
        start_sample: Complex-sample offset where reading starts.
        sample_count: Number of complex IQ samples to read.

    Returns:
        Complex numpy array where each element is `I + jQ`.

    Raises:
        ValueError: If start/count are invalid or not enough IQ data is read.

    Notes:
        The function reads only the requested byte range. This matters because
        one full 2 second AIR recording is large; prediction should not require
        loading the entire file into memory.
    """
    if start_sample < 0:
        raise ValueError("start_sample must be non-negative.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    raw_start_float = 2 * start_sample
    raw_float_count = 2 * sample_count
    byte_offset = raw_start_float * FLOAT32_BYTES

    with file_path.open("rb") as file_handle:
        file_handle.seek(byte_offset)
        raw = np.fromfile(file_handle, dtype=np.float32, count=raw_float_count)

    if raw.size < 2:
        raise ValueError(f"Not enough IQ data could be read from {file_path}")
    if raw.size % 2 != 0:
        raw = raw[:-1]

    i_channel = raw[0::2]
    q_channel = raw[1::2]
    # Reconstruct the analytic baseband signal used by FFT/PSD/spectrogram code.
    return i_channel.astype(np.float32) + 1j * q_channel.astype(np.float32)


def choose_segment_starts(
    total_iq_samples: int,
    segment_samples: int,
    first_start_sample: int,
    segments_per_recording: int,
) -> list[int]:
    """
    Choose segment start positions across a recording.

    Args:
        total_iq_samples: Number of complex samples in the recording.
        segment_samples: Number of complex samples in one analysis segment.
        first_start_sample: Earliest allowed segment start.
        segments_per_recording: Number of evenly spaced segments to select.
            A value less than or equal to zero means use all non-overlapping
            segments.

    Returns:
        Ordered segment start indices in complex samples.

    Raises:
        ValueError: If sample counts are invalid or the first start is outside
            the recording.

    Notes:
        A single 20 ms slice may catch an unusual burst or quiet moment. Evenly
        spaced slices give the classifier repeated observations of the same
        recording. If segments_per_recording is 0 or negative, all non-overlap-
        ping 20 ms segments are used.
    """
    if total_iq_samples <= 0:
        raise ValueError("Recording has no IQ samples.")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive.")
    if first_start_sample >= total_iq_samples:
        raise ValueError("first_start_sample is beyond the recording length.")

    last_valid_start = max(first_start_sample, total_iq_samples - segment_samples)

    if segments_per_recording <= 0:
        starts = list(range(first_start_sample, last_valid_start + 1, segment_samples))
        return starts or [first_start_sample]

    if segments_per_recording == 1 or last_valid_start == first_start_sample:
        return [first_start_sample]

    starts = [
        int(round(value))
        for value in np.linspace(
            first_start_sample,
            last_valid_start,
            num=segments_per_recording,
        )
    ]

    # Rounding can duplicate starts for very short files. Preserve order while
    # removing duplicates so each segment is genuinely different.
    return list(dict.fromkeys(starts))
