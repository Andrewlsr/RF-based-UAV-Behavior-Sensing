"""
Signal-processing primitives for AIR RF behaviour sensing.

This module converts complex IQ segments into RF representations used by
feature extraction. The outputs remain interpretable engineering quantities:
FFT power, Welch PSD, spectrogram power, entropy, and occupied bandwidth.

Notes:
    The functions assume complex baseband IQ samples. Frequency axes are
    offsets from the SDR centre frequency, so a frequency of 0 Hz corresponds
    to the configured receiver centre, not absolute DC in the original RF band.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal


DEFAULT_FFT_SAMPLES = 65_536


def remove_dc(iq_signal: np.ndarray) -> np.ndarray:
    """
    Remove SDR receiver DC offset.

    Args:
        iq_signal: Complex IQ samples.

    Returns:
        IQ samples after subtracting the complex mean.

    Notes:
        SDR captures often contain artificial energy at exactly 0 Hz offset
        because of receiver leakage. Subtracting the complex mean reduces that
        hardware artifact before spectral features are measured.
    """
    return iq_signal - np.mean(iq_signal)


def compute_fft_power(
    iq_signal: np.ndarray,
    sample_rate_hz: float,
    fft_samples: int = DEFAULT_FFT_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a windowed FFT power spectrum.

    Args:
        iq_signal: Complex IQ samples.
        sample_rate_hz: IQ sample rate in Hz.
        fft_samples: Maximum number of samples used in the FFT.

    Returns:
        A tuple `(frequency_hz, power)` where `frequency_hz` is centred around
        0 Hz offset and `power` is squared FFT magnitude.

    Notes:
        FFT converts a short IQ block from time samples into frequency bins.
        Each bin shows energy at a frequency offset around the SDR centre.
        A Hann window is applied to reduce spectral leakage from finite-block
        observation.
    """
    iq_without_dc = remove_dc(iq_signal)
    fft_sample_count = min(int(fft_samples), iq_without_dc.size)
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
    Compute Welch power spectral density.

    Args:
        iq_signal: Complex IQ samples.
        sample_rate_hz: IQ sample rate in Hz.
        nperseg: Welch window length.
        noverlap: Overlap between neighbouring Welch windows.

    Returns:
        A tuple `(frequency_hz, psd)` with two-sided, FFT-shifted frequencies.

    Notes:
        Welch PSD averages many local spectra. Compared with one FFT, it gives
        a steadier view of where the recording keeps its spectral power.
    """
    iq_without_dc = remove_dc(iq_signal)
    nperseg = min(int(nperseg), iq_without_dc.size)
    noverlap = min(int(noverlap), nperseg - 1)

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

    Args:
        iq_signal: Complex IQ samples.
        sample_rate_hz: IQ sample rate in Hz.
        nperseg: Short-time FFT window length.
        noverlap: Overlap between adjacent windows.

    Returns:
        A tuple `(frequency_hz, time_s, spectrogram_power)`.

    Notes:
        A spectrogram is a sequence of short-time FFTs. It shows whether RF
        activity is persistent, bursty, hopping, or changing over the 20 ms
        segment.
    """
    iq_without_dc = remove_dc(iq_signal)
    nperseg = min(int(nperseg), iq_without_dc.size)
    noverlap = min(int(noverlap), nperseg - 1)

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
    """
    Return a mask for bins outside the centre/DC exclusion band.

    Args:
        frequency_hz: FFT-shifted frequency offsets.
        dc_exclusion_hz: Half-width around 0 Hz to exclude.

    Returns:
        Boolean mask where True marks bins used for RF-content features.
    """
    return np.abs(frequency_hz) > dc_exclusion_hz


def spectral_entropy(power: np.ndarray) -> float:
    """
    Normalized spectral entropy.

    Args:
        power: Non-negative spectral power values.

    Returns:
        Entropy normalized to the range `[0, 1]` when power is valid.

    Notes:
        Low entropy means power is concentrated in a few spectral components.
        High entropy means power is spread broadly across frequency.
    """
    power = np.maximum(np.asarray(power, dtype=float), 0.0)
    total_power = np.sum(power)
    if total_power <= 0 or power.size <= 1:
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
    Estimate occupied bandwidth containing a fraction of spectral power.

    Args:
        frequency_hz: Frequency-bin offsets in Hz.
        power: Power associated with each frequency bin.
        fraction: Fraction of total power to include, usually 0.99.

    Returns:
        Frequency span in Hz containing the requested central power fraction.

    Notes:
        Occupied bandwidth measures how wide the active RF energy is. Narrow
        channels, wide Wi-Fi-like blocks, and spread activity can produce
        different bandwidth values.
    """
    if frequency_hz.size == 0:
        return 0.0

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
