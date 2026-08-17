# Final Project Audit Report

Project: RF-based UAV Behaviour Sensing Framework using Explainable Traditional Machine Learning

Audit date: 2026-07-14

## 1. Overall Project Status

**NEEDS MINOR FIXES BEFORE GITHUB RELEASE**

The scientific workflow, leakage-safe validation, engineering prediction pipeline, saved model loading, demo workflow, explainability outputs, and architecture diagram all pass the audit.

The remaining issues are release-packaging issues, not scientific or model-validity problems:

- Generated output CSV/Markdown files contain local machine paths from the audit/demo runs.
- `CODE_INVENTORY_AND_SNAPSHOT.md` is stale and still contains old absolute local paths.
- `__pycache__` folders are present and should not be committed.
- README uses downward arrow symbols; GitHub should render them, but some Windows terminals displayed mojibake during audit. This is a presentation risk only.

## 2. Completed Checks

### Audit 1: Project Structure

**PASS**

The current project structure is suitable for an open-source engineering prototype:

```text
src/
  iq_loader.py
  signal_processing.py
  feature_extraction.py
  preprocessing.py
  model.py
  prediction.py

scripts/
  train_model.py
  predict.py
  run_demo.py
  generate_explainability.py
  create_architecture_diagram.py
  check_reproducibility.py

models/
  classifier.joblib
  scaler.joblib
  feature_config.json

docs/
  system_architecture.png

outputs/
  demo_results.csv
  demo_summary_report.md
  demo_confusion_matrix.png
  explainability/
```

Research scripts `stage1` through `stage7` are preserved as development records. The reusable engineering system is separated into `src/` and `scripts/`.

### Audit 2: Scientific Consistency

**PASS**

The scientific conclusions are internally consistent:

- Stage 4 shows limited direct 12-state separability using simple feature intervals.
- Stage 5 shows flight mode is the strongest RF information source, with secondary interference and interaction effects.
- Stage 6 validates this hierarchy with leakage-safe ML:
  - Flight mode: `80.0%`
  - Interference: `51.7%`
  - 12-state: `41.7%`

README claims match the actual Stage 6 metrics from:

`outputs/stage6_leakage_safe_ml_validation/grouped_cv_overall_metrics.csv`

No audited statement claims distance estimation, localization, trajectory tracking, or autonomous response capability.

### Audit 3: Data Leakage Check

**PASS**

ML validation code groups by original `.dat` recording:

- Stage 6 uses `recording_index` folds.
- The engineering model validation in `src/model.py` also splits by `recording_index`.
- The code checks that train and test recording paths do not overlap.
- Scaler fitting happens inside each training fold during validation.
- Segment predictions are combined by recording-level majority vote.

No filename, file path, mode label, interference label, or state label is included in the numeric feature matrix. The model input uses only `FEATURE_NAMES`.

Prediction mode uses `require_metadata=False`, so a new unknown `.dat` file can be processed without extracting labels for prediction.

### Audit 4: Prediction Pipeline Check

**PASS**

The deployment pipeline is implemented as expected:

```text
.dat file
  -> IQ loading
  -> 20 ms segmentation
  -> RF feature extraction
  -> saved scaler
  -> saved classifier
  -> segment predictions
  -> majority voting
  -> final flight-mode prediction
```

Verified properties:

- Training and prediction use the same `extract_recording_feature_rows` and `extract_segment_features` functions.
- Feature order is saved in `models/feature_config.json`.
- Prediction loads `feature_names` from the saved config.
- `models/scaler.joblib` is loaded and applied before classifier prediction.
- `models/classifier.joblib` loads successfully.
- Confidence is computed as the mean Random Forest class probability across analysed segments.

Confidence is meaningful as a model confidence score, but it should not be interpreted as calibrated probability without calibration analysis.

### Audit 5: Reproducibility Check

**PASS WITH PACKAGING WARNINGS**

Verified by running:

```powershell
python -m compileall src scripts stage1_load_air_iq.py stage4_air_state_analysis.py stage5_factorial_information_analysis.py stage6_leakage_safe_ml_validation.py stage7_train_flight_mode_predictor.py stage7_predict_air_flight_mode.py
python -m pip check
python scripts/check_reproducibility.py --prediction-file path/to/DroneDetect/AIR/CLEAN/AIR_ON/AIR_0000_00.dat
```

Results:

- Python syntax check: **PASS**
- Dependency consistency: **PASS**
- Model loading without retraining: **PASS**
- Prediction without retraining: **PASS**
- Release-facing source files: **PASS** for no hardcoded Windows absolute paths

Packaging warnings:

- Generated outputs contain local paths because they were produced on this machine.
- `CODE_INVENTORY_AND_SNAPSHOT.md` contains stale local paths.
- `__pycache__` folders are present.

These should be removed, ignored, or regenerated before publishing.

### Audit 6: Explainability Check

**PASS**

Reviewed:

- `outputs/explainability/feature_importance.csv`
- `outputs/explainability/feature_importance.png`
- `outputs/explainability/feature_interpretation.md`
- Stage 5 factorial analysis outputs

The explanation is consistent with the statistical findings:

- `spectral_entropy` has mixed flight mode / interference / interaction evidence.
- `rms_power` is consistent with flight-mode evidence but is correctly described as not sufficient alone.
- `spectral_centroid_mhz` has mixed evidence.
- `spectrogram_temporal_variability` has mixed evidence.
- `spectrogram_active_fraction` is strong in model importance and is interpreted as interaction-related.

No contradiction was found between feature importance and factorial analysis. The report correctly warns that permutation importance is interpretability support, not a replacement for grouped validation.

### Audit 7: Documentation Check

**PASS WITH MINOR PRESENTATION WARNING**

README contains:

- Project overview
- Motivation
- Research question
- Methodology
- Dataset description
- Results
- Usage instructions
- Current capabilities and limitations
- Future work

Technical wording is appropriate:

- Uses 鈥淩F-based UAV behaviour sensing鈥?- Avoids claiming distance estimation, tracking, autonomous countermeasures, or localization as current capabilities
- States that the ML classifier is only one component of the passive RF sensing pipeline

Minor warning:

- The README pipeline uses downward arrow symbols. GitHub should render them, but Windows terminal output showed mojibake. For maximum portability, replace arrows with ASCII `->` before release if desired.

### Audit 8: Run Verification

**PASS**

Actually run:

```powershell
python -m compileall src scripts stage1_load_air_iq.py stage4_air_state_analysis.py stage5_factorial_information_analysis.py stage6_leakage_safe_ml_validation.py stage7_train_flight_mode_predictor.py stage7_predict_air_flight_mode.py
python -m pip check
python scripts/check_reproducibility.py --prediction-file path/to/DroneDetect/AIR/CLEAN/AIR_ON/AIR_0000_00.dat
python scripts\predict.py --file path/to/DroneDetect/AIR/CLEAN/AIR_ON/AIR_0000_00.dat
python scripts\run_demo.py path/to/DroneDetect/AIR/CLEAN
```

Results:

- Syntax check: **PASS**
- Requirements check: **PASS**
- Reproducibility checker: **PASS**
- Single-file prediction: **PASS**
- Folder demo: **PASS**

Demo result:

- 15 clean AIR recordings analysed
- 5 ON, 5 HO, 5 FY
- Demo accuracy: `100.0%`
- Output: `outputs/demo_results.csv`

Important interpretation:

The demo result is a demonstration on a selected labeled folder, not the final production performance estimate. The current final production performance estimate is the Round 18 recording-grouped outer cross-validation result: `95.0%` accuracy (`57/60`) for the complete AIR dataset. The earlier Stage 6 result of `80.0%` balanced accuracy remains documented as historical validation of the original production model.

## 3. Remaining Issues

### Issue 1: Generated Outputs Contain Local Paths

Severity: **Minor release-packaging issue**

Files such as generated prediction CSVs and demo reports include local paths from this machine, for example paths under `path/to/DroneDetect/...` and this project workspace.

Recommendation:

- Do not commit generated outputs containing local paths, or regenerate demo outputs with relative paths only.
- Keep `outputs/` mostly ignored except selected final figures/reports that are sanitized.

### Issue 2: Stale Code Inventory Snapshot

Severity: **Minor release-packaging issue**

`CODE_INVENTORY_AND_SNAPSHOT.md` contains old path strings and old code snapshots. It is useful as a private development record but not suitable for public release.

Recommendation:

- Remove it from the GitHub release, or regenerate it after all final cleanup.

### Issue 3: Python Cache Files Present

Severity: **Minor release-packaging issue**

`__pycache__/`, `src/__pycache__/`, and `scripts/__pycache__/` are present.

Recommendation:

- Add `.gitignore`.
- Exclude `__pycache__/`, `*.pyc`, large generated outputs, local datasets, and local model experiments.

### Issue 4: README Arrow Rendering

Severity: **Very minor presentation issue**

README contains downward arrow symbols. GitHub should render UTF-8 Markdown correctly, but one Windows terminal displayed mojibake.

Recommendation:

- Optional: replace arrow symbols with ASCII `->` for maximum terminal portability.

### Issue 5: Saved Model Is Included

Severity: **Policy/project choice**

The model artifacts are present and load correctly. For GitHub, decide whether to publish trained model binaries.

Recommendation:

- If model files are small enough and dataset license permits derived model release, include them.
- Otherwise, document that users should run `python scripts/train_model.py --data-root ...` to regenerate them.

## 4. Recommended Final Actions Before GitHub Release

1. Add `.gitignore` with at least:

```text
__pycache__/
*.pyc
data/
outputs/
*.dat
.venv/
```

Then selectively keep any sanitized figures/reports you want to publish.

2. Remove or exclude:

```text
CODE_INVENTORY_AND_SNAPSHOT.md
__pycache__/
src/__pycache__/
scripts/__pycache__/
outputs/* containing local absolute paths
```

3. Decide whether to publish:

```text
models/classifier.joblib
models/scaler.joblib
models/feature_config.json
```

4. Optionally change README arrow symbols to ASCII `->`.

5. Add a short license file if this will be public. Also check the DroneDetect dataset license before distributing any trained model or derived outputs.

## 5. Final Audit Decision

**NEEDS MINOR FIXES BEFORE GITHUB RELEASE**

The project is scientifically and technically sound for the stated scope:

- Passive RF sensing
- AIR-only DroneDetect analysis
- Explainable RF features
- Traditional ML
- Flight-mode deployment target
- No deep learning
- No exaggerated capability claims

After packaging cleanup, the project should be ready for public release as:

**RF-based UAV Behaviour Sensing Framework using Explainable Traditional Machine Learning**


