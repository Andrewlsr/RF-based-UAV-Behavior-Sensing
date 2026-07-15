# Feature Interpretation Report

## Purpose

This report explains why the AIR flight-mode classifier can predict `ON`,
`HO`, and `FY` from passive RF recordings. The model is a Random Forest trained
on interpretable RF features, not a deep neural network.

## Most Contributing RF Features

- `spectrogram_active_fraction`: permutation drop 0.073, RF importance 0.147, Stage 5 evidence: Mode x interference
- `spectral_entropy`: permutation drop 0.008, RF importance 0.137, Stage 5 evidence: Mixed
- `occupied_bandwidth_mhz`: permutation drop 0.008, RF importance 0.107, Stage 5 evidence: No robust effect
- `fft_peak_frequency_mhz`: permutation drop 0.003, RF importance 0.128, Stage 5 evidence: Flight mode
- `spectrogram_temporal_variability`: permutation drop 0.000, RF importance 0.113, Stage 5 evidence: Mixed

## Comparison With Factorial Analysis

Stage 5 showed that flight mode is the dominant RF information source,
interference is secondary, and a flight-mode x interference interaction exists.
The importance ranking is consistent with that result: the classifier relies on
features describing spectral concentration, activity fraction, frequency
position, bandwidth, and temporal variation rather than on one isolated scalar.

## Required Feature Discussion

- Spectral entropy: `spectral_entropy` has RF importance 0.137 and permutation drop 0.008. Stage 5 conclusion: Mixed.
- RMS power: `rms_power` has RF importance 0.110 and permutation drop 0.000. Stage 5 conclusion: Flight mode.
- Spectral centroid: `spectral_centroid_mhz` has RF importance 0.089 and permutation drop 0.000. Stage 5 conclusion: Mixed.
- Temporal variability: `spectrogram_temporal_variability` has RF importance 0.113 and permutation drop 0.000. Stage 5 conclusion: Mixed.

## Engineering Interpretation

Flight mode changes the RF behaviour because the UAV communication/control link
does not occupy frequency and time in exactly the same way when the drone is
only switched on, hovering, or flying. The strongest evidence comes from
features that describe how RF energy is distributed in frequency and how active
the spectrum is over time.

Power features are useful context, but power alone is not a reliable behaviour
signature because received amplitude can change with geometry, antenna
orientation, and propagation. Spectral and time-frequency features are more
directly tied to waveform structure.

## Caveat

Permutation importance here uses the saved final model and the available AIR
feature matrix. It is an interpretability tool, not a replacement for the
recording-grouped validation result used for performance reporting.
