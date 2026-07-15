# Contributing

Thank you for helping improve this RF-based UAV behaviour sensing framework.

## Project Purpose

This repository focuses on passive RF signal processing, explainable RF feature
engineering, statistical validation, and traditional machine learning for AIR
UAV flight-mode recognition.

Please keep contributions aligned with the current scope:

- RF signal processing and engineering interpretation.
- Explainable feature extraction.
- Leakage-safe validation.
- Traditional machine-learning baselines.
- Clear documentation and reproducibility.

This project intentionally does not use deep learning in its current release.

## Coding Style

- Prefer readable, modular Python.
- Keep public functions documented with docstrings.
- Explain RF signal-processing meaning when adding new analysis code.
- Preserve recording-level grouping when evaluating machine-learning models.
- Avoid hardcoded local dataset paths; use command-line arguments or
  `DRONEDETECT_DATASET_ROOT`.

## Reporting Issues

When opening an issue, include:

- The command you ran.
- Your Python version and operating system.
- Whether model artifacts already existed or you retrained them.
- A short description of the dataset folder structure you used.
- Any traceback or error message.

Please do not upload proprietary or restricted RF recordings unless you have the
right to share them.
