"""
Preprocess RF feature matrices before traditional machine learning.

The feature extractor emits physically meaningful values with different units:
energy, MHz, entropy, and fractions. This module centralizes scaling so
training, validation, and prediction apply the same numerical preprocessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(feature_matrix: np.ndarray) -> StandardScaler:
    """
    Fit a standard scaler to RF features.

    Args:
        feature_matrix: Numeric RF feature matrix with rows as segments and
            columns as features.

    Returns:
        Fitted `StandardScaler`.

    Notes:
        Features use different units: MHz, energy, entropy, and fractions.
        Standardization puts them on comparable numerical scale for models and
        for reproducible saved preprocessing.
    """
    return StandardScaler().fit(feature_matrix)


def scale_features(feature_matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply a fitted scaler to RF feature rows.

    Args:
        feature_matrix: Numeric feature matrix to transform.
        scaler: Fitted `StandardScaler`.

    Returns:
        Scaled feature matrix with the same shape as the input.
    """
    return scaler.transform(feature_matrix)
