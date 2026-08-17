"""
Production temporal optimization layer for AIR UAV flight-mode prediction.

This module is intentionally separate from the original RF production model.

Production architecture:
    Original production RF
        -> baseline prediction
    Additional 40 ms temporal RF
        -> validated FY temporal rule
        -> validated HO temporal correction
        -> final prediction

The original production model remains unchanged.

IMPORTANT
---------
The production temporal rules are loaded exclusively from:

    models/temporal/temporal_rules.json

That JSON file is the single source of truth for the deployed temporal
optimization configuration. No temporal rule values are hard-coded here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = ["ON", "HO", "FY"]

# Repository root:
#   <repo>/src/temporal_optimization.py
REPO_ROOT = Path(__file__).resolve().parents[1]

RULES_PATH = REPO_ROOT / "models" / "temporal" / "temporal_rules.json"


def _load_rule_config() -> dict[str, Any]:
    """
    Load the production temporal rule configuration.

    The JSON file is the only source of truth for temporal rules.
    Fail fast if the production artifact is missing or malformed.
    """
    if not RULES_PATH.exists():
        raise FileNotFoundError(
            "Required temporal production rule artifact not found:\n"
            f"{RULES_PATH}"
        )

    try:
        payload = json.loads(
            RULES_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON in temporal production rule artifact:\n"
            f"{RULES_PATH}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Temporal rule configuration must contain a JSON object."
        )

    if "rules_by_group" not in payload:
        raise ValueError(
            "Temporal rule configuration is missing 'rules_by_group'."
        )

    if "fallback_rules" not in payload:
        raise ValueError(
            "Temporal rule configuration is missing 'fallback_rules'."
        )

    rules_by_group = payload["rules_by_group"]
    fallback_rules = payload["fallback_rules"]

    if not isinstance(rules_by_group, dict):
        raise ValueError(
            "'rules_by_group' must be a JSON object."
        )

    if not isinstance(fallback_rules, dict):
        raise ValueError(
            "'fallback_rules' must be a JSON object."
        )

    required_rule_keys = {
        "fy": {
            "last_n",
            "threshold",
            "delta_threshold",
            "margin_threshold",
        },
        "ho": {
            "last_n",
            "threshold",
            "delta_threshold",
            "margin_threshold",
            "wins_threshold",
            "mean_threshold",
        },
    }

    for group, rules in rules_by_group.items():
        if not isinstance(rules, dict):
            raise ValueError(
                f"Rules for recording group '{group}' must be an object."
            )

        for rule_name, required_keys in required_rule_keys.items():
            if rule_name not in rules:
                raise ValueError(
                    f"Group '{group}' is missing '{rule_name}' rule."
                )

            rule = rules[rule_name]

            if not isinstance(rule, dict):
                raise ValueError(
                    f"Group '{group}' rule '{rule_name}' must be an object."
                )

            missing = required_keys - set(rule.keys())

            if missing:
                raise ValueError(
                    f"Group '{group}' rule '{rule_name}' is missing keys: "
                    f"{sorted(missing)}"
                )

    for rule_name, required_keys in required_rule_keys.items():
        if rule_name not in fallback_rules:
            raise ValueError(
                f"Fallback rules are missing '{rule_name}'."
            )

        rule = fallback_rules[rule_name]

        if not isinstance(rule, dict):
            raise ValueError(
                f"Fallback rule '{rule_name}' must be an object."
            )

        missing = required_keys - set(rule.keys())

        if missing:
            raise ValueError(
                f"Fallback rule '{rule_name}' is missing keys: "
                f"{sorted(missing)}"
            )

    return payload


# Load once when the production module is imported.
#
# This intentionally means that a production prediction run always uses
# the persisted JSON configuration rather than a second hard-coded copy.
RULE_CONFIG = _load_rule_config()

RULES_BY_GROUP: dict[str, dict[str, dict[str, float | int]]] = (
    RULE_CONFIG["rules_by_group"]
)

FALLBACK_RULES: dict[str, dict[str, float | int]] = (
    RULE_CONFIG["fallback_rules"]
)


def _last_mean(probabilities: np.ndarray, n: int) -> float:
    """Return the mean probability over the last n segments."""
    return float(np.mean(probabilities[-n:]))


def apply_fy_rule(
    probabilities: np.ndarray,
    baseline_prediction: str,
    rule: dict[str, float | int],
) -> str:
    """
    Apply the validated FY temporal correction.

    Only ON -> FY correction is permitted.
    """
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
    """
    Apply the validated HO temporal correction.

    Only non-HO -> HO correction is permitted.
    """
    if prediction == "HO":
        return prediction

    p_on = probabilities[:, 0]
    p_ho = probabilities[:, 1]
    p_fy = probabilities[:, 2]

    n = int(rule["last_n"])

    last_ho = _last_mean(p_ho, n)
    first_ho = _last_mean(p_ho[:n], n)

    ho_delta = last_ho - first_ho

    last_on = _last_mean(p_on, n)
    last_fy = _last_mean(p_fy, n)

    ho_margin = last_ho - max(last_on, last_fy)

    mean_ho = float(np.mean(p_ho))

    ho_wins = int(
        np.sum(
            p_ho > np.maximum(p_on, p_fy)
        )
    )

    if (
        last_ho >= float(rule["threshold"])
        and ho_delta >= float(rule["delta_threshold"])
        and ho_margin >= float(rule["margin_threshold"])
        and ho_wins >= int(rule["wins_threshold"])
        and mean_ho >= float(rule["mean_threshold"])
    ):
        return "HO"

    return prediction


def get_rules(
    recording_index: str,
) -> dict[str, dict[str, float | int]]:
    """
    Return the persisted production rules for a recording index.

    Known groups use their validated Round 18 rule.
    Unknown groups use the persisted fallback rule.
    """
    return RULES_BY_GROUP.get(
        str(recording_index),
        FALLBACK_RULES,
    )


def optimize_prediction(
    probabilities: np.ndarray,
    baseline_prediction: str,
    recording_index: str,
) -> tuple[str, str]:
    """
    Apply FY correction, followed by HO correction.

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
        if (
            baseline_prediction == "ON"
            and final_prediction == "FY"
        ):
            return final_prediction, "FY temporal correction"

        if final_prediction == "HO":
            return final_prediction, "HO temporal correction"

    return final_prediction, "No correction"


def save_rules(path: Path) -> None:
    """
    Persist the currently validated production rule configuration.

    This helper is intended for the temporal production training workflow.
    The deployed inference path reads the resulting JSON file directly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            RULE_CONFIG,
            indent=2,
        ),
        encoding="utf-8",
    )