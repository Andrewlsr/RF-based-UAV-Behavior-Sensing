# AIR RF Information Source Analysis

## Question

Does the RF information originate from flight mode, interference conditions,
or the interaction between the two?

## Independent Statistical Units

- Original recordings: 60
- Recordings per 12-state cell: 5
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

`rms_power;fft_peak_frequency_mhz;peak_frequency_mhz;spectral_entropy;spectral_centroid_mhz;spectrogram_temporal_variability`

Interference evidence:

`peak_frequency_mhz;spectral_entropy;spectral_centroid_mhz;spectrogram_temporal_variability`

Interaction evidence:

`spectral_entropy;spectral_centroid_mhz;spectrogram_temporal_variability;spectrogram_active_fraction`

## Feature-Level Conclusions

- `rms_power`: Flight mode (robust effects: flight_mode)
- `signal_energy`: Flight mode (robust effects: flight_mode)
- `fft_peak_frequency_mhz`: Flight mode (robust effects: flight_mode)
- `peak_frequency_mhz`: Mixed (robust effects: flight_mode;interference)
- `occupied_bandwidth_mhz`: No robust effect (robust effects: none)
- `spectral_entropy`: Mixed (robust effects: flight_mode;interference;interaction)
- `spectral_centroid_mhz`: Mixed (robust effects: flight_mode;interference;interaction)
- `spectrogram_temporal_variability`: Mixed (robust effects: flight_mode;interference;interaction)
- `spectrogram_active_fraction`: Mode x interference (robust effects: interaction)

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
