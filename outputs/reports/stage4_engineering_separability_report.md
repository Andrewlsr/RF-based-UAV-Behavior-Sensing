# Stage 4 Engineering Separability Report

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

- Source recordings processed: 60
- Feature rows extracted: 600
- Segments per state: 50 to 50
- Total state pairs checked: 66
- Strong pairs: 0
- Partial pairs: 0
- Overlapping pairs: 66
- State pairs with at least some direct feature evidence: 0.0%

## Result Using Central 80% Ranges

The central 80% range uses the 10th to 90th percentile for each feature. This
is less affected by unusual individual segments.

- Strong pairs: 0
- Partial pairs: 13
- Overlapping pairs: 53
- State pairs with at least some typical-feature evidence: 19.7%

## Best Features for Direct State Separation

- `rms_power` separates 0 of 66 state pairs (0.0%)
- `signal_energy` separates 0 of 66 state pairs (0.0%)
- `fft_peak_frequency_mhz` separates 0 of 66 state pairs (0.0%)
- `peak_frequency_mhz` separates 0 of 66 state pairs (0.0%)
- `occupied_bandwidth_mhz` separates 0 of 66 state pairs (0.0%)

## Best Features for Typical Central-Range Separation

- `spectral_entropy` separates 10 of 66 state pairs (15.2%)
- `rms_power` separates 2 of 66 state pairs (3.0%)
- `signal_energy` separates 2 of 66 state pairs (3.0%)
- `spectrogram_active_fraction` separates 2 of 66 state pairs (3.0%)
- `fft_peak_frequency_mhz` separates 0 of 66 state pairs (0.0%)

## Features That Vary Most Across Flight Modes

- `spectral_entropy`
- `rms_power`
- `fft_peak_frequency_mhz`
- `spectrogram_temporal_variability`
- `signal_energy`

## Features That Vary Most Across Interference Conditions

- `spectral_entropy`
- `spectrogram_active_fraction`
- `spectral_centroid_mhz`
- `peak_frequency_mhz`
- `spectrogram_temporal_variability`

## Engineering Conclusion

The current RF feature set shows limited direct separability. More segment sampling and richer features are needed before claiming all 12 states are separable.

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
