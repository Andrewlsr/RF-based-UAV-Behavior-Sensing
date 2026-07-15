# Stage 7 Flight-Mode Prediction Report

## Purpose

Train and save a deployable traditional machine-learning predictor for AIR
flight mode:

- `ON`
- `HO`
- `FY`

## Why Flight Mode First

Stage 6 leakage-safe validation showed that flight mode is the strongest
generalizable RF target:

- Best model: Random Forest
- Recording-level balanced accuracy: about 80%
- Chance level: 33.3%

## Model

- Model type: Random Forest
- Training rows: 600 RF feature rows
- Source recordings: 60 AIR `.dat` files
- Segments per recording: 10
- Segment duration: 20 ms
- Input features: Stage 4 RF feature set

## Prediction Workflow

For a new AIR `.dat` recording:

1. Select 10 evenly spaced 20 ms segments.
2. Extract the same RF features used during training.
3. Predict one flight mode per segment.
4. Majority-vote the segment predictions.
5. Report mean class probabilities.

## Sanity Checks

Known files were tested after training:

- `AIR_0000_00.dat`: predicted `ON`
- `AIR_0001_00.dat`: predicted `HO`
- `AIR_0010_00.dat`: predicted `FY`

All three matched the mode encoded in the dataset path/name.

## Important Limitation

This final model is trained on all available AIR feature rows after validation.
Its unbiased performance estimate comes from Stage 6, not from these sanity
checks.

