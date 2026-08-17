# RF-based UAV Behaviour Sensing using Explainable Machine Learning

## 1. Overview

I developed an RF-based UAV behaviour recognition framework that extracts behaviour-related features from passively received radio-frequency signals and identifies UAV operational states, providing sensing-layer information for future airspace monitoring, risk assessment, and autonomous regulation systems.

The implementation uses the DroneDetect AIR recordings to build an engineering prototype that converts raw IQ data into interpretable RF features and predicts UAV flight mode using traditional machine learning.

The final deployment target is flight-mode recognition:

- `ON`: switched on
- `HO`: hovering
- `FY`: flying

The project intentionally uses RF signal processing, explainable feature engineering, statistical analysis, and traditional machine learning. It does not use CNNs, transformers, or end-to-end deep learning.

## 2. Motivation

Passive RF sensing can support airspace monitoring because UAVs emit communication and control signals during operation. A behaviour-recognition layer can provide useful perception data for later systems such as UAV detection support, monitoring dashboards, risk assessment, and regulatory decision support.

This prototype focuses on answering one engineering question:

Can UAV operating behaviour be inferred from passively received RF IQ recordings?

## 3. Methodology

The system pipeline is:

```text
IQ Signal
    -> Signal Processing
    -> Feature Extraction
    -> Statistical Analysis
    -> Machine Learning
    -> Behaviour Recognition
```

Signal-processing stages:

- IQ loading and reconstruction from interleaved float32 samples
- DC offset removal to reduce receiver centre-frequency artifacts
- FFT analysis for frequency-domain inspection
- Welch PSD analysis for stable spectral comparison
- Spectrogram analysis for time-frequency behaviour

Extracted RF features:

- RMS power
- Signal energy
- FFT peak frequency
- PSD peak frequency
- Occupied bandwidth
- Spectral entropy
- Spectral centroid
- Spectrogram activity fraction
- Spectrogram temporal variability

These features describe received signal strength, where energy sits in frequency, how concentrated or spread the spectrum is, and how RF activity changes over time.

## 4. Dataset

The project uses the AIR subset of the DroneDetect dataset.

Dataset source:

- DroneDetect Dataset on IEEE DataPort:
  <https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine>
- DOI: <https://dx.doi.org/10.21227/5jjj-1m32>

If you use this project, please cite the DroneDetect dataset in addition to this
software repository.

Dataset properties:

- File format: `.dat`
- Sample format: interleaved float32 IQ samples
- IQ layout: `I0, Q0, I1, Q1, ...`
- Sample rate: `60 MHz`
- Centre frequency: `2.4375 GHz`
- Bandwidth: approximately `28 MHz`
- Recording duration: approximately `2 seconds`

Flight modes:

- `ON`: switched on
- `HO`: hovering
- `FY`: flying

Interference conditions:

- `00`: clean
- `01`: Bluetooth
- `10`: Wi-Fi
- `11`: Bluetooth + Wi-Fi

The current public-release code is configurable. Place the dataset under `data/` or pass a dataset path with `--data-root`. You can also set:

```powershell
$env:DRONEDETECT_DATASET_ROOT = "path/to/DroneDetect/AIR"
```

## 5. Results

### Final production result

The final production pipeline was evaluated on all 60 AIR recordings using recording-grouped outer cross-validation with five held-out groups and inner grouped out-of-fold rule selection. No held-out recording was used to train its outer-fold model, and no all-data self-test was used.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|---:|
| Original production RF | 91.67% | 91.67% | 91.61% | 55/60 |
| + FY temporal rule | 93.33% | 93.33% | 93.27% | 56/60 |
| **Final FY + HO temporal optimization** | **95.00%** | **95.00%** | **94.97%** | **57/60** |

Final class performance:

| Class | Correct |
|---|---:|
| ON | 20/20 (100.00%) |
| HO | 17/20 (85.00%) |
| FY | 20/20 (100.00%) |

Final confusion matrix:

| True / Pred | ON | HO | FY |
|---|---:|---:|---:|
| ON | 20 | 0 | 0 |
| HO | 0 | 17 | 3 |
| FY | 0 | 0 | 20 |

The temporal optimization changed two recording-level predictions, corrected both baseline errors, and introduced zero harmful changes.

### Historical engineering validation

The earlier Stage 6 analysis remains documented as historical validation of the original engineering model:

| Task | Balanced accuracy |
|---|---:|
| Flight mode: `ON` / `HO` / `FY` | `80.0%` |
| Interference: `00` / `01` / `10` / `11` | `51.7%` |
| Full 12-state tag | `41.7%` |

These historical results should not be confused with the current final temporal production result above.
## 6. Usage

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the final flight-mode predictor:

```powershell
python scripts/train_model.py --data-root path/to/DroneDetect/AIR
```

Predict one AIR recording:

```powershell
python scripts/predict.py --file path/to/example.dat
```

Run a folder-level demo:

```powershell
python scripts/run_demo.py path/to/DroneDetect/AIR/CLEAN
```

Generate explainability outputs:

```powershell
python scripts/generate_explainability.py
```

Create the architecture diagram:

```powershell
python scripts/create_architecture_diagram.py
```

Run release reproducibility checks:

```powershell
python scripts/check_reproducibility.py --prediction-file path/to/example.dat
```

Saved model artifacts:

- `models/classifier.joblib`
- `models/scaler.joblib`
- `models/feature_config.json`

Key outputs:

- `outputs/reports/`
- `outputs/final_figures/`
- `outputs/explainability/feature_importance.csv`
- `outputs/explainability/feature_importance.png`
- `outputs/explainability/feature_interpretation.md`
- `docs/system_architecture.png`

## 7. Current Capabilities and Limitations

Current capabilities:

The system can:

- process UAV RF IQ recordings
- reconstruct complex IQ samples from interleaved float32 data
- extract behaviour-related RF features
- identify AIR flight mode: `ON` / `HO` / `FY`
- provide prediction confidence
- save segment-level prediction tables for engineering inspection

Current limitations:

The system currently does not:

- estimate UAV distance
- localize UAV position
- track UAV trajectory
- identify exact geographic location
- provide autonomous countermeasure decisions
- guarantee generalization to unseen drone models, receivers, environments, or collection geometries without additional validation

## 8. Future Work

Localization:

- RSSI modelling
- Angle of Arrival
- Multi-antenna sensing
- TDOA

Behaviour monitoring:

- anomaly detection
- unknown RF pattern detection
- trajectory estimation

Real-time deployment:

- SDR hardware integration
- continuous RF monitoring
- streaming feature extraction
- online confidence monitoring

## Project Positioning

This repository should be understood as an RF-based UAV behaviour sensing framework using explainable traditional machine learning. The ML classifier is one component in a larger passive RF signal-processing pipeline.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this framework in research or engineering work, please cite it as:

Markdown:

> RF-based UAV Behaviour Sensing using Explainable Machine Learning. 2026.
> RF signal processing and explainable traditional machine learning framework
> for UAV behaviour recognition. MIT License.
> <https://github.com/Andrewlsr/RF-based-UAV-Behavior-Sensing>.

BibTeX:

```bibtex
@software{rf_uav_behaviour_sensing_2026,
  title = {RF-based UAV Behaviour Sensing using Explainable Machine Learning},
  year = {2026},
  license = {MIT},
  url = {https://github.com/Andrewlsr/RF-based-UAV-Behavior-Sensing},
  note = {RF signal processing and explainable traditional machine learning framework for UAV behaviour recognition}
}
```

Please also cite the dataset used in this project:

Markdown:

> DroneDetect Dataset: Radio Frequency Dataset for Unmanned Aerial System
> (UAS) Signals. IEEE DataPort. DOI:
> <https://dx.doi.org/10.21227/5jjj-1m32>.

BibTeX:

```bibtex
@dataset{dronedetect_dataset,
  title = {DroneDetect Dataset: Radio Frequency Dataset for Unmanned Aerial System (UAS) Signals},
  publisher = {IEEE DataPort},
  doi = {10.21227/5jjj-1m32},
  url = {https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine}
}
```
