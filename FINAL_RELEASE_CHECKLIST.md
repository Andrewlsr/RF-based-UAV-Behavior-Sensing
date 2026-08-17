# Final Release Checklist

## Repository Cleanup

PASS

The repository has been cleaned for public GitHub release without changing RF
algorithms, feature definitions, model settings, validation logic, or prediction
behaviour.

Completed cleanup:

- Added `.gitignore` for Python caches, virtual environments, local datasets,
  IDE settings, temporary files, and generated scratch outputs.
- Added MIT `LICENSE`.
- Added `CONTRIBUTING.md`.
- Added `CITATION.cff`.
- Added official DroneDetect IEEE DataPort dataset source and DOI to README.
- Updated README usage examples to use portable paths.
- Fixed README pipeline arrow rendering.
- Removed stale local-path development snapshot.
- Removed bytecode cache files and cache directories.
- Removed raw/debug/generated prediction outputs containing local dataset paths.
- Consolidated useful release artifacts into:
  - `outputs/explainability/`
  - `outputs/final_figures/`
  - `outputs/reports/`

## Removed Files

Removed from release:

- `CODE_INVENTORY_AND_SNAPSHOT.md`
- `__pycache__/`
- `src/__pycache__/`
- `scripts/__pycache__/`
- `outputs/engineering_predictions/`
- `outputs/engineering_prediction_system/`
- `outputs/stage4_air_state_analysis/`
- `outputs/stage4_debug/`
- `outputs/stage4_debug_segments/`
- `outputs/stage5_factorial_information_analysis/`
- `outputs/stage6_leakage_safe_ml_validation/`
- `outputs/stage7_predictions/`
- `outputs/stage7_prediction_model/`
- generated demo files:
  - `outputs/demo_results.csv`
  - `outputs/demo_summary_report.md`
  - `outputs/demo_confusion_matrix.png`

Useful reports and figures were retained by copying selected final artifacts
into the release-facing output folders before cleanup.

## Remaining Warnings

- Demo and prediction scripts still generate local output files when run. These
  outputs are ignored or cleaned after verification because they may include
  user-specific input paths.

## Hygiene Scan

PASS

Content-only scan results:

```text
MATCH_COUNT=0
```

No release-facing text files contain:

- absolute Windows drive paths
- Windows user-profile paths
- local username markers
- private IDE path markers

Cache scan:

```text
No __pycache__, .pytest_cache, .venv, venv, or env directories found.
```

## Verification Results

### Python Syntax Check

Command:

```powershell
python -m compileall src scripts
```

Result: PASS

### Python Dependency Check

Command:

```powershell
python -m pip check
```

Result: PASS

Observed output:

```text
No broken requirements found.
```

### Reproducibility Check

Command:

```powershell
python scripts/check_reproducibility.py
```

Result: PASS

Observed checks:

- Python package imports: PASS
- Model artifact loading: PASS
- Prediction without retraining: PASS, skipped because no prediction file was
  supplied to this portable release command
- Output path organization: PASS
- No hardcoded local absolute paths in release-facing files: PASS

Report saved at:

```text
outputs/reports/reproducibility_check.md
```

### Prediction Verification

Command used with a local AIR validation recording:

```powershell
python scripts/predict.py --file path/to/DroneDetect/AIR/CLEAN/AIR_ON/AIR_0000_00.dat
```

Result: PASS

Observed output:

```text
Predicted flight mode: Switched on
Segments analysed: 10
Votes: ON=10, HO=0, FY=0
Confidence: 76.9%
Prediction matches label: True
```

The generated local prediction CSV was removed after verification.

### Demo Verification

Command used with a local AIR clean-folder validation set:

```powershell
python scripts/run_demo.py path/to/DroneDetect/AIR/CLEAN
```

Result: PASS

Observed output:

```text
AIR recordings found: 15
Overall demo accuracy: 100.0%
```

Generated local demo outputs were removed after verification.

## Final Production Validation — Round 18

The final production pipeline was validated on all 60 AIR recordings using recording-grouped outer cross-validation with five held-out groups and inner grouped out-of-fold rule selection.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|---:|
| Original production RF | 91.67% | 91.67% | 91.61% | 55/60 |
| + FY temporal rule | 93.33% | 93.33% | 93.27% | 56/60 |
| **Final FY + HO temporal optimization** | **95.00%** | **95.00%** | **94.97%** | **57/60** |

Final class performance:

- ON: 20/20 (100.00%)
- HO: 17/20 (85.00%)
- FY: 20/20 (100.00%)

The temporal optimization corrected two baseline errors and introduced zero harmful changes.

Validation was recording-grouped and out-of-sample. No all-data self-test was used, and no model artifact was modified during validation.

The earlier Stage 6 metrics remain historical validation results for the original production model.
## GitHub Release Readiness

READY FOR RELEASE

The code, documentation, MIT license, citation metadata, saved models,
explainability outputs, architecture diagram, final reports, and final figures
are organized for public release.
