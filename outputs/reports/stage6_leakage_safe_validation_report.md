# Leakage-Safe Traditional ML Validation

## Validation Design

- 600 RF feature rows from 60 original `.dat` recordings
- 10 segments per recording
- Five grouped folds
- Every fold holds out one complete recording from each of the 12 states
- Scaling is fitted on training folds only
- No segment from a test recording appears in training

## Main Results

- `flight_mode`: best model `RandomForest`, balanced accuracy 0.800 versus chance 0.333, macro F1 0.795
- `interference`: best model `RBFSVM`, balanced accuracy 0.517 versus chance 0.250, macro F1 0.486
- `state_12`: best model `RBFSVM`, balanced accuracy 0.417 versus chance 0.083, macro F1 0.381

## Cross-Condition Generalization

- `flight_mode_holdout_interference`: best mean recording-level balanced accuracy 0.650 using `RandomForest`; worst held-out condition 0.533; chance 0.333
- `interference_holdout_mode`: best mean recording-level balanced accuracy 0.283 using `RandomForest`; worst held-out condition 0.250; chance 0.250

## Engineering Conclusion

- Flight-mode information generalizes across interference conditions, so it is a genuine and relatively stable RF source.
- Interference classification is close to chance when a complete flight mode is unseen. The interference signature is therefore strongly mode-dependent rather than independent.
- The 12-state task is well above its 8.3% chance level, so useful joint
  mode/interference information exists, but the current features do not fully
  separate all 12 states.
- Overall, the most defensible interpretation is: **flight mode is the primary
  generalizable RF source; interference information is weaker and largely
  expressed through its interaction with flight mode.**

## How To Interpret The Results

- Performance near chance means the extracted features do not support that
  classification task reliably.
- Performance clearly above chance means usable RF information exists.
- Strong grouped-CV performance but weak cross-condition performance means the
  model is learning state-specific combinations rather than a factor that
  generalizes independently.
- Recording-level results are the primary engineering result. Segment-level
  results show how reliable a single 20 ms decision is.

## Important Limitation

These are fixed baseline models, not tuned final models. Hyperparameter tuning,
if added later, must use a nested grouped validation procedure.
