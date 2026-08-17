"""
Train the additional 40 ms temporal RF model used by production optimization.

IMPORTANT:
- The original models/classifier.joblib is NOT modified.
- The original 20 ms production model remains the baseline.
- This creates models/temporal/ as a second model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import FeatureConfig, FEATURE_NAMES, build_air_feature_rows, feature_rows_to_matrix
from src.iq_loader import FLIGHT_MODE_LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "temporal")
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING ADDITIONAL 40 MS TEMPORAL PRODUCTION MODEL")
    print("=" * 80)
    print(f"Dataset: {args.data_root}")
    print("Original production model will NOT be modified.")

    config = FeatureConfig(
        segment_ms=40.0,
        segments_per_recording=10,
    )

    rows = build_air_feature_rows(args.data_root, config)

    if not rows:
        raise RuntimeError("No feature rows were extracted.")

    X = feature_rows_to_matrix(rows, FEATURE_NAMES)
    y = np.asarray([str(row["mode_code"]) for row in rows])

    scaler = __import__("src.preprocessing", fromlist=["fit_scaler"]).fit_scaler(X)
    X_scaled = __import__("src.preprocessing", fromlist=["scale_features"]).scale_features(X, scaler)

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_scaled, y)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_dir / "classifier.joblib")
    joblib.dump(scaler, args.model_dir / "scaler.joblib")

    config_payload = {
        **config.to_dict(),
        "target": "flight_mode",
        "feature_names": FEATURE_NAMES,
        "class_order": ["ON", "HO", "FY"],
        "label_mapping": FLIGHT_MODE_LABELS,
        "training_feature_rows": len(rows),
        "training_recordings": len({str(row["file_path"]) for row in rows}),
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
        },
        "purpose": "Additional 40 ms temporal model; original production RF remains unchanged.",
    }
    (args.model_dir / "feature_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    from src.temporal_optimization import save_rules
    save_rules(args.model_dir / "temporal_rules.json")

    print()
    print("Saved:")
    print(args.model_dir / "classifier.joblib")
    print(args.model_dir / "scaler.joblib")
    print(args.model_dir / "feature_config.json")
    print(args.model_dir / "temporal_rules.json")
    print()
    print("DONE. Original models/ artifacts were NOT modified.")


if __name__ == "__main__":
    main()
