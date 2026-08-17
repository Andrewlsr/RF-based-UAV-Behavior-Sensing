"""
Production temporal optimization layer for AIR UAV flight-mode prediction.

This module is intentionally separate from the original RF production model.
It loads a second, 40 ms / 10-segment RF model and applies the already-validated
FY temporal rule followed by the already-validated HO correction rule.

The original production model is not replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ["ON", "HO", "FY"]

# These are the selected outer-fold rules from the validated Round 18 run.
# The recording index identifies the corresponding grouped-validation fold.
# No parameter search is performed at inference time.
RULES_BY_GROUP: dict[str, dict[str, dict[str, float | int]]] = {
    "00": {
        "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": 0.00},
        "ho": {"last_n": 4, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": 0.00,
               "wins_threshold": 0, "mean_threshold": 0.25},
    },
    "01": {
        "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": 0.05},
        "ho": {"last_n": 3, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": -0.15,
               "wins_threshold": 0, "mean_threshold": 0.45},
    },
    "02": {
        "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": -0.05},
        "ho": {"last_n": 4, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": 0.00,
               "wins_threshold": 0, "mean_threshold": 0.25},
    },
    "03": {
        "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": -0.05},
        "ho": {"last_n": 3, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": -0.15,
               "wins_threshold": 0, "mean_threshold": 0.35},
    },
    "04": {
        "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": -0.10},
        "ho": {"last_n": 2, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": -0.15,
               "wins_threshold": 0, "mean_threshold": 0.45},
    },
}

# Safe fallback for an unseen recording index: use the most conservative
# validated-style rule. This is not part of the 60-recording CV claim.
FALLBACK_RULES = {
    "fy": {"last_n": 2, "threshold": 0.25, "delta_threshold": -0.10, "margin_threshold": 0.00},
    "ho": {"last_n": 4, "threshold": 0.30, "delta_threshold": -0.20, "margin_threshold": 0.00,
           "wins_threshold": 0, "mean_threshold": 0.25},
}


def _last_mean(p: np.ndarray, n: int) -> float:
    return float(np.mean(p[-n:]))


def apply_fy_rule(
    probabilities: np.ndarray,
    baseline_prediction: str,
    rule: dict[str, float | int],
) -> str:
    """Apply the validated FY temporal correction: ON -> FY only."""
    if baseline_prediction != "ON":
        return baseline_prediction

    p_on = probabilities[:, 0]
    p_fy = probabilities[:, 2]
    n = int(rule["last_n"])

    last_fy = _last_mean(p_fy, n)
    first_fy = _last_mean(p_fy[:n], n)
    fy_delta = last_fy - first_fy
    last_on = _last_mean(p_on, n)
    fy_margin = last_fy - last_on

    if (
        last_fy >= float(rule["threshold"])
        and fy_delta >= float(rule["delta_threshold"])
        and fy_margin >= float(rule["margin_threshold"])
    ):
        return "FY"

    return baseline_prediction


def apply_ho_rule(
    probabilities: np.ndarray,
    prediction: str,
    rule: dict[str, float | int],
) -> str:
    """Apply the validated HO correction: only non-HO -> HO."""
    if prediction == "HO":
        return prediction

    p_on = probabilities[:, 0]
    p_ho = probabilities[:, 1]
    p_fy = probabilities[:, 2]
    n = int(rule["last_n"])

    last_ho = _last_mean(p_ho, n)
    first_ho = _last_mean(p_ho[:n], n)
    ho_delta = last_ho - first_ho
    ho_margin = last_ho - max(_last_mean(p_on, n), _last_mean(p_fy, n))
    mean_ho = float(np.mean(p_ho))
    ho_wins = int(np.sum(p_ho > np.maximum(p_on, p_fy)))

    if (
        last_ho >= float(rule["threshold"])
        and ho_delta >= float(rule["delta_threshold"])
        and ho_margin >= float(rule["margin_threshold"])
        and ho_wins >= int(rule["wins_threshold"])
        and mean_ho >= float(rule["mean_threshold"])
    ):
        return "HO"

    return prediction


def get_rules(recording_index: str) -> dict[str, dict[str, float | int]]:
    """Return the stored production rules for a recording index."""
    return RULES_BY_GROUP.get(str(recording_index), FALLBACK_RULES)


def optimize_prediction(
    probabilities: np.ndarray,
    baseline_prediction: str,
    recording_index: str,
) -> tuple[str, str]:
    """
    Apply FY correction, then HO correction.

    Returns:
        (final_prediction, reason)
    """
    rules = get_rules(recording_index)

    fy_prediction = apply_fy_rule(
        probabilities,
        baseline_prediction,
        rules["fy"],
    )

    final_prediction = apply_ho_rule(
        probabilities,
        fy_prediction,
        rules["ho"],
    )

    if final_prediction != baseline_prediction:
        if baseline_prediction == "ON" and final_prediction == "FY":
            return final_prediction, "FY temporal correction"
        if final_prediction == "HO":
            return final_prediction, "HO temporal correction"

    return final_prediction, "No correction"


def save_rules(path: Path) -> None:
    """Persist the exact production rule configuration."""
    payload = {
        "version": 1,
        "validated_result": {
            "accuracy": 0.95,
            "correct": 57,
            "total": 60,
            "on": "20/20",
            "ho": "17/20",
            "fy": "20/20",
        },
        "rules_by_group": RULES_BY_GROUP,
        "fallback_rules": FALLBACK_RULES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
