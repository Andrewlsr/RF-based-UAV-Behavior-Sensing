"""
Stages 1-3: Load, inspect, run FFT, and plot spectrogram for one AIR recording.

The DroneDetect .dat files store raw radio samples as interleaved float32
values. The format is:

    I0, Q0, I1, Q1, I2, Q2, ...

where:
    I = in-phase component
    Q = quadrature component

Together, one I value and one Q value form one complex IQ sample:

    IQ[n] = I[n] + j * Q[n]

In RF signal processing, the complex IQ representation keeps both amplitude
and phase information. This is important because a drone control/video signal
is not only described by how strong it is, but also by how its phase and
frequency content change over time.

For the beginning of this project we only inspect the AIR class data stored
under:

    data

You can either pass a specific AIR .dat file path, or let the script search
for the first AIR .dat file under the dataset directory.

Practical note:
    A full 2 second DroneDetect recording is very large. It can contain about
    1.2e8 complex IQ samples, stored as about 2.4e8 float32 values. Loading the
    whole file can use almost 1 GB of memory.

    For these early inspection stages, the script counts the full file using
    its file size, then reads only the first block of IQ samples needed for the
    time-domain plot and FFT. This keeps the script fast and beginner-friendly
    while still using NumPy to read the binary IQ data.

FFT note:
    The time-domain IQ plot shows how the received signal changes sample by
    sample. The FFT converts that same complex IQ signal into the frequency
    domain, showing where RF energy is located inside the 60 MHz sampled band.

Spectrogram note:
    A spectrogram repeats the FFT over short time windows. It shows how the
    RF spectrum changes over time, which is important for drone signals and
    interference that may hop, burst, or change activity during a recording.
"""

import argparse
import os
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing required package: numpy\n\n"
        "Install it into the same Python interpreter you use to run this script:\n"
        "    python -m pip install numpy\n"
    ) from error

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing required package: matplotlib\n\n"
        "Install it into the same Python interpreter you use to run this script:\n"
        "    python -m pip install matplotlib\n\n"
        "Reason: matplotlib is used to plot the first 5000 I samples and "
        "the first 5000 Q samples."
    ) from error

try:
    from scipy import fft as scipy_fft
    from scipy import signal
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing required package: scipy\n\n"
        "Install it into the same Python interpreter you use to run this script:\n"
        "    python -m pip install scipy\n\n"
        "Reason: scipy.fft is used for FFT and scipy.signal.spectrogram is "
        "used for time-frequency analysis."
    ) from error


# Root folder provided for the AIR dataset.
# The script searches below this folder when no exact .dat file is supplied.
DEFAULT_DATASET_ROOT = Path(os.environ.get("DRONEDETECT_DATASET_ROOT", "data"))

# Number of I and Q samples to show in the beginner visual inspection plot.
DEFAULT_PLOT_SAMPLE_COUNT = 5000

# Number of complex IQ samples to use for FFT analysis.
# More samples give finer frequency resolution, but take more computation.
DEFAULT_FFT_SAMPLE_COUNT = 65_536

# Number of complex IQ samples to use for spectrogram analysis.
# At 60 MHz, 1,200,000 samples equals 20 ms, which matches the common
# DroneDetect segment length mentioned in the project description.
DEFAULT_SPECTROGRAM_SAMPLE_COUNT = 1_200_000

# Spectrogram window length. At 60 MHz, 4096 samples is about 68.3 microseconds.
# This gives enough time resolution to see bursts while still giving a useful
# frequency resolution of about 14.6 kHz per FFT bin.
DEFAULT_SPECTROGRAM_NPERSEG = 4096

# 50% overlap is a common engineering starting point for spectrograms.
DEFAULT_SPECTROGRAM_OVERLAP = 2048

# DroneDetect recordings were captured at 60 MHz sample rate.
# This means the digital IQ sequence can represent a 60 MHz-wide baseband span:
# approximately -30 MHz to +30 MHz around the SDR centre frequency.
DEFAULT_SAMPLE_RATE_HZ = 60_000_000.0

# SDR centre frequency used for DroneDetect recordings.
# FFT plots use frequency offset, but this value lets us explain the absolute
# RF frequency too.
DEFAULT_CENTER_FREQUENCY_HZ = 2_437_500_000.0

# When summarising the strongest FFT peak, ignore a small band around 0 Hz.
# Energy exactly around 0 Hz is often receiver DC / LO leakage rather than the
# drone signal we want to interpret.
DEFAULT_DC_EXCLUSION_HZ = 250_000.0

# A float32 value occupies 4 bytes on disk.
FLOAT32_BYTES = np.dtype(np.float32).itemsize

# Small positive value used before log10 so we never take log10(0).
EPSILON = 1e-12

# DroneDetect interference code used in the file names.
INTERFERENCE_LABELS = {
    "00": "clean signal, no Bluetooth or Wi-Fi interference",
    "01": "Bluetooth interference only",
    "10": "Wi-Fi interference only",
    "11": "Bluetooth and Wi-Fi interference together",
}

INTERFERENCE_FOLDER_TO_CODE = {
    "CLEAN": "00",
    "BLUE": "01",
    "WIFI": "10",
    "BOTH": "11",
}

INTERFERENCE_INPUT_TO_CODE = {
    "clean": "00",
    "blue": "01",
    "bluetooth": "01",
    "wifi": "10",
    "both": "11",
    "00": "00",
    "01": "01",
    "10": "10",
    "11": "11",
}

# Common DroneDetect flight-mode codes. Some dataset copies expose the mode
# through folders named ON/HO/FY, while some encode it numerically in filenames.
FLIGHT_MODE_LABELS = {
    "ON": "switched on",
    "HO": "hovering",
    "FY": "flying",
    "00": "switched on",
    "01": "hovering",
    "10": "flying",
}

MODE_INPUT_TO_CODE = {
    "on": "ON",
    "switched-on": "ON",
    "ho": "HO",
    "hovering": "HO",
    "fy": "FY",
    "flying": "FY",
    "00": "00",
    "01": "01",
    "10": "10",
}

INTERFERENCE_SORT_ORDER = {"00": 0, "01": 1, "10": 2, "11": 3}
MODE_SORT_ORDER = {"ON": 0, "00": 0, "HO": 1, "01": 1, "FY": 2, "10": 2}


def normalise_interference_filter(interference: str | None) -> str | None:
    """Convert a user-friendly interference argument to a DroneDetect code."""
    if interference is None:
        return None

    key = interference.strip().lower()
    if key not in INTERFERENCE_INPUT_TO_CODE:
        valid_values = ", ".join(sorted(INTERFERENCE_INPUT_TO_CODE))
        raise ValueError(
            f"Unknown interference filter '{interference}'. "
            f"Use one of: {valid_values}"
        )

    return INTERFERENCE_INPUT_TO_CODE[key]


def normalise_mode_filter(mode: str | None) -> str | None:
    """Convert a user-friendly flight-mode argument to a DroneDetect code."""
    if mode is None:
        return None

    key = mode.strip().lower()
    if key not in MODE_INPUT_TO_CODE:
        valid_values = ", ".join(sorted(MODE_INPUT_TO_CODE))
        raise ValueError(f"Unknown mode filter '{mode}'. Use one of: {valid_values}")

    return MODE_INPUT_TO_CODE[key]


def parse_recording_metadata(file_path: Path) -> dict[str, str | None]:
    """
    Decode model, interference, mode, and file index from a DroneDetect path.

    This uses dataset metadata from the path and filename. It does not infer
    the class by looking at the signal itself.
    """
    upper_name = file_path.name.upper()
    upper_parts = [part.upper() for part in file_path.parts]

    metadata = {
        "model": upper_name.split("_", maxsplit=1)[0],
        "interference_code": None,
        "mode_code": None,
        "file_index": None,
    }

    filename_fields = upper_name.replace(".DAT", "").split("_")
    if len(filename_fields) >= 2 and len(filename_fields[1]) >= 4:
        metadata["interference_code"] = filename_fields[1][:2]
        metadata["mode_code"] = filename_fields[1][2:4]
    if len(filename_fields) >= 3:
        metadata["file_index"] = filename_fields[2]

    for folder_name, interference_code in INTERFERENCE_FOLDER_TO_CODE.items():
        if folder_name in upper_parts:
            metadata["interference_code"] = interference_code
            break

    # Folder names can be more explicit than file codes, so prefer them when
    # they are available in the path. In this dataset, folders may be named
    # like AIR_ON, AIR_HO, and AIR_FY rather than only ON, HO, and FY.
    for folder_mode_code in ("ON", "HO", "FY"):
        if any(
            part == folder_mode_code or part.endswith(f"_{folder_mode_code}")
            for part in upper_parts
        ):
            metadata["mode_code"] = folder_mode_code
            break

    return metadata


def sort_air_files_for_analysis(file_paths: list[Path]) -> list[Path]:
    """
    Sort AIR files in a useful analysis order.

    Instead of relying on alphabetical folder order, this puts baseline clean
    recordings first, then Bluetooth, Wi-Fi, and both-interference recordings.
    Within each group it orders switched-on, hovering, then flying.
    """

    def sort_key(file_path: Path) -> tuple[int, int, str]:
        """
        Build a stable ordering key from AIR recording metadata.

        Parameters
        ----------
        file_path : pathlib.Path
            AIR `.dat` recording path.

        Returns
        -------
        tuple[int, int, str]
            Interference rank, flight-mode rank, and path string. The path is
            included as the final element so files with the same state remain
            deterministically ordered.
        """
        metadata = parse_recording_metadata(file_path)
        interference_rank = INTERFERENCE_SORT_ORDER.get(
            metadata["interference_code"],
            99,
        )
        mode_rank = MODE_SORT_ORDER.get(metadata["mode_code"], 99)
        return interference_rank, mode_rank, str(file_path)

    return sorted(file_paths, key=sort_key)


def find_air_dat_files(dataset_root: Path) -> list[Path]:
    """
    Find AIR .dat recordings inside the dataset folder.

    DroneDetect file names include a short drone/model identifier. For this
    first stage we only want AIR data, so we search for files whose names start
    with AIR and end with .dat, for example:

        AIR_0000_00.dat

    The recursive search is useful because the dataset may be organised by
    interference group and flight mode folders.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}\n"
            "Check that the E: drive is connected and the path is correct."
        )

    print(f"Searching for AIR .dat files under: {dataset_root}")
    return sort_air_files_for_analysis(list(dataset_root.rglob("AIR*.dat")))


def choose_input_file(
    file_path: str | None,
    dataset_root: Path,
    interference: str | None,
    mode: str | None,
    recording_index: int,
) -> Path:
    """
    Decide which .dat file to load.

    If the user gives a file path, we use that exact file. Otherwise, we search
    the AIR dataset and pick the first AIR .dat file so the script can run with
    a simple command during early exploration.
    """
    if file_path is not None:
        selected_file = Path(file_path)
        if not selected_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {selected_file}")
        if selected_file.suffix.lower() != ".dat":
            raise ValueError(f"Input file must be a .dat file: {selected_file}")
        if not selected_file.name.upper().startswith("AIR"):
            print(
                "Warning: this file name does not start with 'AIR'. "
                "For this project stage we are focusing on AIR data only."
            )
        return selected_file

    target_interference_code = normalise_interference_filter(interference)
    target_mode_code = normalise_mode_filter(mode)

    air_files = find_air_dat_files(dataset_root)
    if not air_files:
        raise FileNotFoundError(
            f"No AIR .dat files were found under: {dataset_root}"
        )

    filtered_files = []
    for air_file in air_files:
        metadata = parse_recording_metadata(air_file)
        if (
            target_interference_code is not None
            and metadata["interference_code"] != target_interference_code
        ):
            continue
        if (
            target_mode_code is not None
            and metadata["mode_code"] != target_mode_code
        ):
            continue
        filtered_files.append(air_file)

    if not filtered_files:
        raise FileNotFoundError(
            "No AIR .dat files matched the requested filters. "
            f"Interference={interference}, mode={mode}"
        )

    if recording_index < 0 or recording_index >= len(filtered_files):
        raise IndexError(
            f"--recording-index {recording_index} is out of range. "
            f"{len(filtered_files)} file(s) matched the current filters."
        )

    print("No input file was supplied.")
    print(f"Using AIR file #{recording_index} after sorting/filtering: {filtered_files[recording_index]}")
    print(f"Total AIR .dat files found under dataset root: {len(air_files)}")
    print(f"Files matching current filters: {len(filtered_files)}")
    return filtered_files[recording_index]


def describe_recording_from_path(file_path: Path) -> None:
    """
    Print the AIR recording type, interference group, and flight mode.

    RF/dataset meaning:
        The same drone can look different in the RF domain depending on the
        environment and state:

        - Interference changes the spectrum because Bluetooth/Wi-Fi signals
          share the 2.4 GHz band.
        - Flight mode changes the drone's radio activity because switched-on,
          hovering, and flying states can produce different control/video link
          behaviour.

    This function uses the filename and folder names as metadata. It does not
    inspect the waveform itself to infer the mode.
    """
    metadata = parse_recording_metadata(file_path)
    model = metadata["model"]
    interference_code = metadata["interference_code"]
    mode_code = metadata["mode_code"]
    file_index = metadata["file_index"]

    print("\n--- Recording metadata from path/name ---")
    print(f"Selected file: {file_path}")
    print(f"Drone/model label: {model}")

    if interference_code in INTERFERENCE_LABELS:
        print(
            f"Interference group: {interference_code} "
            f"({INTERFERENCE_LABELS[interference_code]})"
        )
    else:
        print("Interference group: could not decode from filename")

    if mode_code in FLIGHT_MODE_LABELS:
        print(f"Flight mode: {mode_code} ({FLIGHT_MODE_LABELS[mode_code]})")
    else:
        print("Flight mode: could not decode from filename/path")

    if file_index is not None:
        print(f"Recording index inside this condition: {file_index}")


def count_raw_floats_from_file_size(file_path: Path) -> int:
    """
    Count how many float32 values are stored in the full .dat file.

    RF meaning:
        The file stores I and Q as separate float values. Two raw floats make
        one complex IQ sample. Counting bytes lets us know the full recording
        length without loading the whole recording into memory.
    """
    file_size_bytes = file_path.stat().st_size

    if file_size_bytes % FLOAT32_BYTES != 0:
        raise ValueError(
            "File size is not divisible by 4 bytes. "
            "This does not look like clean float32 IQ data."
        )

    return file_size_bytes // FLOAT32_BYTES


def load_interleaved_float32_iq_preview(
    file_path: Path,
    iq_sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a preview of interleaved float32 values and convert to complex IQ.

    The raw .dat file is arranged as pairs:

        raw[0] = I sample 0
        raw[1] = Q sample 0
        raw[2] = I sample 1
        raw[3] = Q sample 1

    Therefore:
        I channel = every even-indexed value
        Q channel = every odd-indexed value

    The complex IQ signal is then:
        iq = I + 1j * Q
    """
    # Each complex IQ sample needs two stored float values: one I and one Q.
    raw_float_count_to_read = 2 * iq_sample_count

    # np.fromfile reads the binary file directly into a NumPy array.
    # dtype=np.float32 is essential because the DroneDetect samples are saved
    # as 32-bit floating point values, not text and not integers.
    raw_float_data = np.fromfile(
        file_path,
        dtype=np.float32,
        count=raw_float_count_to_read,
    )

    # Every complex sample needs exactly two floats: one I and one Q.
    # An odd number of floats would mean the final sample is incomplete.
    if raw_float_data.size % 2 != 0:
        raise ValueError(
            "The file contains an odd number of float32 values. "
            "Interleaved IQ data should contain I/Q pairs."
        )

    # I samples are stored at positions 0, 2, 4, ...
    i_channel = raw_float_data[0::2]

    # Q samples are stored at positions 1, 3, 5, ...
    q_channel = raw_float_data[1::2]

    # Combine I and Q into a complex-valued signal.
    # This gives one complex number per RF sample.
    iq_signal = i_channel + 1j * q_channel

    return i_channel, q_channel, iq_signal


def print_basic_iq_information(
    raw_float_count: int,
    iq_signal: np.ndarray,
    sample_rate_hz: float,
) -> None:
    """
    Print simple checks that confirm the file was interpreted correctly.

    raw float count:
        Number of float32 values stored in the full file.

    IQ sample count:
        Number of complex RF samples in the full file after pairing I and Q.

    first 10 IQ samples:
        A quick sanity check showing the first complex samples as I + jQ.
    """
    print("\n--- Basic IQ file information ---")
    print(f"Raw float count: {raw_float_count}")
    print(f"IQ sample count: {raw_float_count // 2}")
    print(f"Full recording duration: {(raw_float_count // 2) / sample_rate_hz:.3f} seconds")
    print(f"Preview block duration: {iq_signal.size / sample_rate_hz * 1e3:.3f} ms")
    print("\nFirst 10 IQ samples:")

    for index, sample in enumerate(iq_signal[:10]):
        print(f"  IQ[{index}] = {sample.real:.8f} + j({sample.imag:.8f})")


def print_fft_explanation(
    sample_rate_hz: float,
    center_frequency_hz: float,
    fft_sample_count: int,
) -> None:
    """
    Explain the RF meaning of the FFT before plotting it.

    This is intentionally printed by the script because the goal of the project
    is not just to produce plots, but to understand what the plots mean.
    """
    frequency_resolution_hz = sample_rate_hz / fft_sample_count
    half_span_mhz = sample_rate_hz / 2 / 1e6
    fft_duration_ms = fft_sample_count / sample_rate_hz * 1e3
    center_frequency_ghz = center_frequency_hz / 1e9

    print("\n--- FFT / RF spectrum explanation ---")
    print(
        "FFT means Fast Fourier Transform. It converts the complex IQ signal "
        "from time domain into frequency domain."
    )
    print(
        "Physically, this tells us how much RF energy exists at each frequency "
        "offset inside the captured baseband."
    )
    print(
        f"With a {sample_rate_hz / 1e6:.1f} MHz sample rate, the FFT frequency "
        f"axis spans about -{half_span_mhz:.1f} MHz to +{half_span_mhz:.1f} MHz "
        "around the SDR centre frequency."
    )
    print(
        f"The centre frequency is {center_frequency_ghz:.4f} GHz, so an FFT "
        "offset of 0 MHz corresponds to that RF frequency."
    )
    print(
        f"Using {fft_sample_count} IQ samples gives frequency-bin spacing of "
        f"about {frequency_resolution_hz:.1f} Hz and observes "
        f"{fft_duration_ms:.3f} ms of signal."
    )
    print(
        "Narrow peaks can indicate tones or narrowband interferers. Wider "
        "raised regions can indicate spread-spectrum, Wi-Fi-like activity, or "
        "a drone link occupying bandwidth."
    )
    print(
        "The dB plot is relative power, not calibrated dBm. It is useful for "
        "comparing frequency regions inside the same recording."
    )


def compute_fft_spectrum(
    iq_signal: np.ndarray,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the FFT magnitude and relative power spectrum of complex IQ data.

    RF meaning of each step:
        1. DC removal:
           Subtract the mean IQ value. SDR recordings often contain a strong
           artificial component at 0 Hz caused by receiver DC offset rather
           than the drone signal itself.

        2. Windowing:
           Multiply by a Hann window before the FFT. This reduces spectral
           leakage, where energy from one frequency smears into nearby bins
           because we only observe a finite block of samples.

        3. FFT:
           Convert the time-domain complex IQ block into frequency bins.
           Each FFT bin represents energy at a frequency offset.

        4. FFT shift:
           Move 0 Hz to the middle of the plot so negative frequencies appear
           on the left and positive frequencies appear on the right. This is
           the standard view for complex baseband IQ signals.
    """
    if iq_signal.size < 4:
        raise ValueError("At least four IQ samples are required for FFT analysis.")

    # Convert to complex64/complex128 behaviour explicitly through NumPy.
    # The subtraction removes receiver DC offset at 0 Hz.
    iq_without_dc = iq_signal - np.mean(iq_signal)

    # A Hann window improves the visual spectrum by reducing edge discontinuity
    # between the start and end of the selected IQ block.
    window = np.hanning(iq_without_dc.size)
    window_sum = np.sum(window)
    if window_sum <= 0:
        raise ValueError("FFT window has zero gain. Use more IQ samples.")

    windowed_iq = iq_without_dc * window

    # scipy_fft.fft computes the discrete frequency content of the IQ block.
    fft_values = scipy_fft.fft(windowed_iq)
    fft_values = scipy_fft.fftshift(fft_values)

    # Create the matching frequency axis using the 60 MHz sample rate.
    # For complex IQ, this gives both negative and positive baseband offsets.
    frequency_hz = scipy_fft.fftfreq(iq_without_dc.size, d=1.0 / sample_rate_hz)
    frequency_hz = scipy_fft.fftshift(frequency_hz)

    # Magnitude tells us the amplitude of each FFT bin.
    # Dividing by the window sum keeps values at a manageable scale.
    magnitude = np.abs(fft_values) / window_sum

    # Power is proportional to magnitude squared.
    # We normalize by the peak so the strongest bin is 0 dB, making the plot
    # easy to interpret even without calibrated receiver gain.
    power = magnitude**2
    peak_power = np.max(power)
    if peak_power <= 0:
        power_db = np.full_like(power, -120.0)
    else:
        # Clip the lower bound before log10. This keeps the strongest bin at
        # exactly 0 dB while avoiding log10(0) for very weak bins.
        relative_power = np.maximum(power / peak_power, EPSILON)
        power_db = 10.0 * np.log10(relative_power)

    return frequency_hz, magnitude, power_db


def find_strongest_non_dc_index(
    frequency_hz: np.ndarray,
    values: np.ndarray,
    dc_exclusion_hz: float,
) -> int | None:
    """
    Find the strongest frequency bin outside the centre/DC region.

    The centre of an SDR baseband recording can contain receiver artifacts.
    For interpretation plots, the strongest non-centre bin is often more useful
    than the absolute strongest bin.
    """
    non_dc_mask = np.abs(frequency_hz) > dc_exclusion_hz
    if not np.any(non_dc_mask):
        return None

    non_dc_indices = np.where(non_dc_mask)[0]
    return int(non_dc_indices[int(np.argmax(values[non_dc_mask]))])


def smooth_for_plot(values: np.ndarray, window_length: int) -> np.ndarray:
    """
    Smooth a noisy trace for plotting only.

    This does not change the FFT or spectrogram calculation. It only adds a
    cleaner visual guide on top of the raw spectrum so the dominant RF regions
    are easier to see.
    """
    if window_length <= 1 or values.size < window_length:
        return values

    if window_length % 2 == 0:
        window_length += 1

    kernel = np.ones(window_length) / window_length
    return np.convolve(values, kernel, mode="same")


def print_fft_summary(
    frequency_hz: np.ndarray,
    power_db: np.ndarray,
    center_frequency_hz: float,
    dc_exclusion_hz: float,
) -> None:
    """
    Print a compact interpretation summary for the FFT plot.

    The strongest bin may be at 0 Hz because SDR hardware can leak energy at
    the centre frequency. For drone/interference interpretation, the strongest
    non-DC peak is often more useful.
    """
    strongest_index = int(np.argmax(power_db))
    strongest_offset_hz = frequency_hz[strongest_index]
    strongest_absolute_hz = center_frequency_hz + strongest_offset_hz

    strongest_non_dc_index = find_strongest_non_dc_index(
        frequency_hz,
        power_db,
        dc_exclusion_hz,
    )
    if strongest_non_dc_index is not None:
        strongest_non_dc_offset_hz = frequency_hz[strongest_non_dc_index]
        strongest_non_dc_absolute_hz = center_frequency_hz + strongest_non_dc_offset_hz
        strongest_non_dc_power_db = power_db[strongest_non_dc_index]
    else:
        strongest_non_dc_offset_hz = None
        strongest_non_dc_absolute_hz = None
        strongest_non_dc_power_db = None

    noise_floor_db = float(np.median(power_db))

    print("\n--- FFT summary ---")
    print(
        "Strongest FFT bin: "
        f"{strongest_offset_hz / 1e6:.3f} MHz offset "
        f"({strongest_absolute_hz / 1e9:.6f} GHz absolute RF), "
        f"{power_db[strongest_index]:.2f} dB relative power"
    )
    if strongest_non_dc_offset_hz is not None:
        print(
            f"Strongest non-centre bin outside +/-{dc_exclusion_hz / 1e3:.0f} kHz: "
            f"{strongest_non_dc_offset_hz / 1e6:.3f} MHz offset "
            f"({strongest_non_dc_absolute_hz / 1e9:.6f} GHz absolute RF), "
            f"{strongest_non_dc_power_db:.2f} dB relative power"
        )
    print(f"Approximate median spectrum level: {noise_floor_db:.2f} dB")


def print_spectrogram_explanation(
    sample_rate_hz: float,
    spectrogram_sample_count: int,
    nperseg: int,
    noverlap: int,
) -> None:
    """
    Explain the RF meaning of a spectrogram before plotting it.

    A single FFT is one frequency snapshot. A spectrogram is many FFTs over
    consecutive short windows, which lets us see whether energy is constant,
    bursty, hopping, or changing over time.
    """
    segment_duration_us = nperseg / sample_rate_hz * 1e6
    hop_size = nperseg - noverlap
    hop_duration_us = hop_size / sample_rate_hz * 1e6
    total_duration_ms = spectrogram_sample_count / sample_rate_hz * 1e3
    frequency_resolution_hz = sample_rate_hz / nperseg

    print("\n--- Spectrogram / time-frequency explanation ---")
    print(
        "A spectrogram splits the IQ signal into short overlapping time "
        "windows, computes an FFT for each window, and stacks those FFTs over time."
    )
    print(
        "Physically, this shows when RF energy appears and at which frequency "
        "offset it appears."
    )
    print(
        f"This spectrogram uses {spectrogram_sample_count} IQ samples "
        f"({total_duration_ms:.3f} ms of recording)."
    )
    print(
        f"Each FFT window is {nperseg} samples "
        f"({segment_duration_us:.2f} microseconds), with a hop of {hop_size} "
        f"samples ({hop_duration_us:.2f} microseconds)."
    )
    print(
        f"The approximate frequency spacing is {frequency_resolution_hz:.1f} Hz "
        "per bin."
    )
    print(
        "Bluetooth interference may appear as narrow, short-lived energy that "
        "moves or hops across frequency."
    )
    print(
        "Wi-Fi interference often appears as wider MHz-scale blocks of energy, "
        "sometimes bursty because packets are transmitted in time."
    )
    print(
        "Drone signals may show persistent or structured energy bands, and "
        "their activity can differ between switched-on, hovering, and flying modes."
    )


def compute_spectrogram(
    iq_signal: np.ndarray,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    nperseg: int = DEFAULT_SPECTROGRAM_NPERSEG,
    noverlap: int = DEFAULT_SPECTROGRAM_OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a two-sided spectrogram for complex IQ data.

    RF meaning of each step:
        1. DC removal:
           Subtract the mean IQ value to reduce receiver centre-frequency
           bias before the time-frequency calculation.

        2. Short-time FFT:
           scipy.signal.spectrogram performs many FFTs over short windows.
           Each column of the output is one local spectrum.

        3. Two-sided complex-IQ frequency axis:
           Complex baseband IQ contains negative and positive frequency
           offsets, so we keep the full two-sided spectrum and shift 0 Hz to
           the middle for interpretation.

        4. dB conversion:
           Power is converted to relative dB so weak and strong components can
           be seen on the same plot. The strongest spectrogram cell is 0 dB.
    """
    if iq_signal.size < 4:
        raise ValueError("At least four IQ samples are required for spectrogram analysis.")

    adjusted_nperseg = min(nperseg, iq_signal.size)
    if adjusted_nperseg < 4:
        raise ValueError("Spectrogram window must contain at least four samples.")

    adjusted_noverlap = min(noverlap, adjusted_nperseg - 1)

    iq_without_dc = iq_signal - np.mean(iq_signal)

    frequency_hz, time_seconds, spectrogram_power = signal.spectrogram(
        iq_without_dc,
        fs=sample_rate_hz,
        window="hann",
        nperseg=adjusted_nperseg,
        noverlap=adjusted_noverlap,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )

    frequency_hz = scipy_fft.fftshift(frequency_hz)
    spectrogram_power = scipy_fft.fftshift(spectrogram_power, axes=0)

    peak_power = np.max(spectrogram_power)
    if peak_power <= 0:
        spectrogram_db = np.full_like(spectrogram_power, -120.0)
    else:
        relative_power = np.maximum(spectrogram_power / peak_power, EPSILON)
        spectrogram_db = 10.0 * np.log10(relative_power)

    return frequency_hz, time_seconds, spectrogram_db


def estimate_spectrogram_focus_band(
    frequency_hz: np.ndarray,
    spectrogram_db: np.ndarray,
    dc_exclusion_hz: float,
    half_width_hz: float = 4_000_000.0,
) -> tuple[float, float, float | None]:
    """
    Choose a readable zoom band for the spectrogram.

    The full 60 MHz view is useful for context, but it can be visually sparse.
    This helper averages power over time, ignores the centre/DC region, and
    zooms around the strongest non-centre frequency band.
    """
    average_power = np.mean(10.0 ** (spectrogram_db / 10.0), axis=1)
    strongest_index = find_strongest_non_dc_index(
        frequency_hz,
        average_power,
        dc_exclusion_hz,
    )

    if strongest_index is None:
        return float(np.min(frequency_hz)), float(np.max(frequency_hz)), None

    focus_center_hz = float(frequency_hz[strongest_index])
    lower_hz = max(float(np.min(frequency_hz)), focus_center_hz - half_width_hz)
    upper_hz = min(float(np.max(frequency_hz)), focus_center_hz + half_width_hz)
    return lower_hz, upper_hz, focus_center_hz


def plot_spectrogram(
    frequency_hz: np.ndarray,
    time_seconds: np.ndarray,
    spectrogram_db: np.ndarray,
    center_frequency_hz: float,
    dc_exclusion_hz: float,
) -> None:
    """
    Plot a time-frequency spectrogram in relative dB.

    How to read the plot:
        - X axis is time inside the selected recording block.
        - Y axis is frequency offset from the centre frequency.
        - Colour is relative power in dB. Brighter colours are stronger RF
          energy. Darker colours are weaker energy or noise floor.
    """
    if time_seconds.size > 0:
        time_ms = (time_seconds - time_seconds[0]) * 1e3
    else:
        time_ms = time_seconds * 1e3

    frequency_mhz = frequency_hz / 1e6
    center_frequency_ghz = center_frequency_hz / 1e9

    # Clip the colour scale for readability. A single very bright cell should
    # not make the rest of the spectrogram unreadably dark.
    display_vmax = min(0.0, max(-30.0, float(np.percentile(spectrogram_db, 99.8))))
    display_vmin = max(-95.0, display_vmax - 70.0)

    focus_lower_hz, focus_upper_hz, focus_center_hz = estimate_spectrogram_focus_band(
        frequency_hz,
        spectrogram_db,
        dc_exclusion_hz,
    )
    focus_mask = (frequency_hz >= focus_lower_hz) & (frequency_hz <= focus_upper_hz)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(
        f"Spectrogram / Time-Frequency View, centre = {center_frequency_ghz:.4f} GHz",
        fontsize=14,
    )

    mesh = axes[0].pcolormesh(
        time_ms,
        frequency_mhz,
        spectrogram_db,
        shading="auto",
        cmap="magma",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[0].set_title("Full 60 MHz Overview (colour scale clipped for contrast)")
    axes[0].set_ylabel("Frequency offset (MHz)")
    axes[0].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    zoom_title = "Zoom Around Strongest Non-Centre Time-Frequency Activity"
    if focus_center_hz is not None:
        zoom_title += f" ({focus_center_hz / 1e6:.2f} MHz)"

    axes[1].pcolormesh(
        time_ms,
        frequency_mhz[focus_mask],
        spectrogram_db[focus_mask, :],
        shading="auto",
        cmap="magma",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[1].set_title(zoom_title)
    axes[1].set_xlabel("Time inside selected block (ms)")
    axes[1].set_ylabel("Frequency offset (MHz)")
    axes[1].set_ylim(focus_lower_hz / 1e6, focus_upper_hz / 1e6)
    if focus_lower_hz <= 0.0 <= focus_upper_hz:
        axes[1].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    if focus_center_hz is not None:
        axes[1].axhline(
            focus_center_hz / 1e6,
            color="cyan",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
        )

    colorbar = figure.colorbar(mesh, ax=axes, pad=0.02)
    colorbar.set_label("Power relative to strongest cell (dB)")


def plot_iq_time_domain(
    i_channel: np.ndarray,
    q_channel: np.ndarray,
    sample_count: int = DEFAULT_PLOT_SAMPLE_COUNT,
) -> None:
    """
    Plot the first few I and Q samples in the time domain.

    This plot does not yet show frequency content. It is a first visual check
    of the waveform values coming from the SDR recording.

    RF meaning:
        - I is one baseband component of the received signal.
        - Q is the same signal shifted by 90 degrees in phase.
        - Looking at I and Q separately helps reveal clipping, DC offset,
          silence, abnormal scaling, or other loading/data issues.
    """
    samples_to_plot = min(sample_count, i_channel.size, q_channel.size)
    sample_index = np.arange(samples_to_plot)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    plt.plot(sample_index, i_channel[:samples_to_plot], color="tab:blue", linewidth=0.8)
    plt.title(f"First {samples_to_plot} I Samples")
    plt.xlabel("Sample index")
    plt.ylabel("I amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(sample_index, q_channel[:samples_to_plot], color="tab:orange", linewidth=0.8)
    plt.title(f"First {samples_to_plot} Q Samples")
    plt.xlabel("Sample index")
    plt.ylabel("Q amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_fft_spectrum(
    frequency_hz: np.ndarray,
    magnitude: np.ndarray,
    power_db: np.ndarray,
    center_frequency_hz: float,
    dc_exclusion_hz: float,
) -> None:
    """
    Plot readable FFT magnitude and power spectrum views.

    How to read the plots:
        - X axis is frequency offset from the SDR centre frequency.
          With a 2.4375 GHz centre frequency, 0 MHz means 2.4375 GHz.
          +5 MHz means about 2.4425 GHz, and -5 MHz means about 2.4325 GHz.

        - Magnitude spectrum shows the linear FFT amplitude.

        - Power spectrum in dB makes weak and strong components easier to
          compare. A value near 0 dB is the strongest component in this FFT
          block. Values like -20 dB or -40 dB are weaker relative components.

        - The shaded region around 0 MHz is the DC/centre area. A spike here
          is often an SDR receiver artifact, so the zoomed view focuses on the
          strongest non-centre activity.
    """
    frequency_mhz = frequency_hz / 1e6
    center_frequency_ghz = center_frequency_hz / 1e9
    dc_exclusion_mhz = dc_exclusion_hz / 1e6
    strongest_non_dc_index = find_strongest_non_dc_index(
        frequency_hz,
        power_db,
        dc_exclusion_hz,
    )

    non_dc_mask = np.abs(frequency_hz) > dc_exclusion_hz
    non_dc_magnitude = magnitude[non_dc_mask] if np.any(non_dc_mask) else magnitude
    magnitude_ymax = float(np.percentile(non_dc_magnitude, 99.8) * 1.4)
    if strongest_non_dc_index is not None:
        magnitude_ymax = max(magnitude_ymax, float(magnitude[strongest_non_dc_index] * 1.25))
    if magnitude_ymax <= 0:
        magnitude_ymax = float(np.max(magnitude) * 1.05)

    smoothed_power_db = smooth_for_plot(power_db, window_length=151)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    figure.suptitle(
        f"FFT Spectrum, centre = {center_frequency_ghz:.4f} GHz",
        fontsize=14,
    )

    axes[0].plot(frequency_mhz, magnitude, color="tab:green", linewidth=0.8)
    axes[0].set_title("Magnitude Spectrum (linear scale, DC spike clipped for readability)")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_ylim(0.0, magnitude_ymax)

    axes[1].plot(
        frequency_mhz,
        power_db,
        color="0.65",
        linewidth=0.5,
        alpha=0.45,
        label="raw dB spectrum",
    )
    axes[1].plot(
        frequency_mhz,
        smoothed_power_db,
        color="tab:red",
        linewidth=1.0,
        label="smoothed guide",
    )
    axes[1].set_title("Power Spectrum Overview (relative dB)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].set_ylim(-95, 5)
    axes[1].legend(loc="upper right")

    if strongest_non_dc_index is not None:
        peak_frequency_mhz = frequency_mhz[strongest_non_dc_index]
        zoom_half_width_mhz = 4.0
        zoom_mask = (
            (frequency_mhz >= peak_frequency_mhz - zoom_half_width_mhz)
            & (frequency_mhz <= peak_frequency_mhz + zoom_half_width_mhz)
        )
        axes[2].plot(
            frequency_mhz[zoom_mask],
            power_db[zoom_mask],
            color="0.65",
            linewidth=0.5,
            alpha=0.45,
        )
        axes[2].plot(
            frequency_mhz[zoom_mask],
            smoothed_power_db[zoom_mask],
            color="tab:red",
            linewidth=1.1,
        )
        axes[2].axvline(
            peak_frequency_mhz,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            label=f"strongest non-centre: {peak_frequency_mhz:.2f} MHz",
        )
        axes[2].set_xlim(peak_frequency_mhz - zoom_half_width_mhz, peak_frequency_mhz + zoom_half_width_mhz)
        axes[2].set_title("Zoom Around Strongest Non-Centre Activity")
        axes[2].legend(loc="upper right")
    else:
        axes[2].plot(frequency_mhz, smoothed_power_db, color="tab:red", linewidth=1.0)
        axes[2].set_title("Zoom unavailable: no non-centre peak found")

    axes[2].set_xlabel("Frequency offset from centre frequency (MHz)")
    axes[2].set_ylabel("Power (dB)")
    axes[2].set_ylim(-95, 5)

    for axis in axes:
        axis.axvspan(
            -dc_exclusion_mhz,
            dc_exclusion_mhz,
            color="gray",
            alpha=0.14,
            label="DC/centre artifact region",
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.grid(True, alpha=0.25)

    axes[1].set_xlabel("Frequency offset from centre frequency (MHz)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))


def parse_arguments() -> argparse.Namespace:
    """
    Parse beginner-friendly command-line options.

    Example 1: let the script find the first AIR file automatically
        python stage1_load_air_iq.py

    Example 2: load a specific AIR file
        python stage1_load_air_iq.py --file "path\\to\\AIR_0000_00.dat"

    Example 3: use a different dataset root
        python stage1_load_air_iq.py --data-root "path\\to\\DroneDetect\\DATASET"
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load one DroneDetect AIR .dat file and plot I/Q, FFT, and "
            "spectrogram analysis."
        )
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional path to a specific AIR .dat file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root folder containing AIR .dat files. Default: data, or the "
            "DRONEDETECT_DATASET_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--interference",
        type=str,
        default=None,
        help=(
            "Optional AIR interference filter: clean/00, blue/bluetooth/01, "
            "wifi/10, or both/11."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional flight-mode filter: on/00, ho/hovering/01, or fy/flying/10.",
    )
    parser.add_argument(
        "--recording-index",
        type=int,
        default=0,
        help="Which matching file to load after sorting/filtering. Default: 0",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=DEFAULT_PLOT_SAMPLE_COUNT,
        help="Number of I and Q samples to plot. Default: 5000",
    )
    parser.add_argument(
        "--fft-samples",
        type=int,
        default=DEFAULT_FFT_SAMPLE_COUNT,
        help="Number of IQ samples to use for FFT. Default: 65536",
    )
    parser.add_argument(
        "--spectrogram-samples",
        type=int,
        default=DEFAULT_SPECTROGRAM_SAMPLE_COUNT,
        help=(
            "Number of IQ samples to use for spectrogram. "
            "Default: 1200000, equal to 20 ms at 60 MHz"
        ),
    )
    parser.add_argument(
        "--spectrogram-nperseg",
        type=int,
        default=DEFAULT_SPECTROGRAM_NPERSEG,
        help="FFT window length used inside the spectrogram. Default: 4096",
    )
    parser.add_argument(
        "--spectrogram-overlap",
        type=int,
        default=DEFAULT_SPECTROGRAM_OVERLAP,
        help="Overlap between neighbouring spectrogram windows. Default: 2048",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="IQ sample rate in Hz. Default: 60000000 for 60 MHz",
    )
    parser.add_argument(
        "--center-frequency",
        type=float,
        default=DEFAULT_CENTER_FREQUENCY_HZ,
        help="SDR centre frequency in Hz. Default: 2437500000 for 2.4375 GHz",
    )
    parser.add_argument(
        "--dc-exclusion-hz",
        type=float,
        default=DEFAULT_DC_EXCLUSION_HZ,
        help=(
            "Frequency span around 0 Hz ignored when reporting strongest "
            "non-centre FFT peak. Default: 250000"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the first-stage IQ loading and plotting workflow."""
    args = parse_arguments()

    if args.plot_samples <= 0:
        raise ValueError("--plot-samples must be greater than zero.")
    if args.fft_samples <= 0:
        raise ValueError("--fft-samples must be greater than zero.")
    if args.spectrogram_samples <= 0:
        raise ValueError("--spectrogram-samples must be greater than zero.")
    if args.spectrogram_nperseg < 4:
        raise ValueError("--spectrogram-nperseg must be at least 4.")
    if args.spectrogram_overlap < 0:
        raise ValueError("--spectrogram-overlap cannot be negative.")
    if args.spectrogram_overlap >= args.spectrogram_nperseg:
        raise ValueError("--spectrogram-overlap must be smaller than --spectrogram-nperseg.")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero.")
    if args.center_frequency <= 0:
        raise ValueError("--center-frequency must be greater than zero.")
    if args.dc_exclusion_hz < 0:
        raise ValueError("--dc-exclusion-hz cannot be negative.")

    input_file = choose_input_file(
        args.file,
        args.data_root,
        args.interference,
        args.mode,
        args.recording_index,
    )
    print(f"\nLoading file: {input_file}")
    describe_recording_from_path(input_file)

    raw_float_count = count_raw_floats_from_file_size(input_file)
    if raw_float_count % 2 != 0:
        raise ValueError(
            "The file contains an odd number of float32 values. "
            "Interleaved IQ data should contain complete I/Q pairs."
        )

    iq_samples_to_read = max(
        args.plot_samples,
        args.fft_samples,
        args.spectrogram_samples,
    )
    print(
        f"Reading the first {iq_samples_to_read} IQ samples for time-domain "
        "preview, FFT analysis, and spectrogram analysis. "
        "The full file is counted from its size on disk."
    )

    i_channel, q_channel, iq_signal = load_interleaved_float32_iq_preview(
        input_file,
        iq_samples_to_read,
    )

    print_basic_iq_information(raw_float_count, iq_signal, args.sample_rate)
    plot_iq_time_domain(i_channel, q_channel, args.plot_samples)

    fft_iq_signal = iq_signal[: args.fft_samples]
    print_fft_explanation(
        args.sample_rate,
        args.center_frequency,
        fft_iq_signal.size,
    )
    frequency_hz, magnitude, power_db = compute_fft_spectrum(
        fft_iq_signal,
        args.sample_rate,
    )
    print_fft_summary(
        frequency_hz,
        power_db,
        args.center_frequency,
        args.dc_exclusion_hz,
    )
    plot_fft_spectrum(
        frequency_hz,
        magnitude,
        power_db,
        args.center_frequency,
        args.dc_exclusion_hz,
    )

    spectrogram_iq_signal = iq_signal[: args.spectrogram_samples]
    actual_spectrogram_nperseg = min(
        args.spectrogram_nperseg,
        spectrogram_iq_signal.size,
    )
    actual_spectrogram_overlap = min(
        args.spectrogram_overlap,
        actual_spectrogram_nperseg - 1,
    )
    print_spectrogram_explanation(
        args.sample_rate,
        spectrogram_iq_signal.size,
        actual_spectrogram_nperseg,
        actual_spectrogram_overlap,
    )
    spectrogram_frequency_hz, spectrogram_time_s, spectrogram_db = compute_spectrogram(
        spectrogram_iq_signal,
        args.sample_rate,
        args.spectrogram_nperseg,
        args.spectrogram_overlap,
    )
    plot_spectrogram(
        spectrogram_frequency_hz,
        spectrogram_time_s,
        spectrogram_db,
        args.center_frequency,
        args.dc_exclusion_hz,
    )

    plt.show()


if __name__ == "__main__":
    main()
