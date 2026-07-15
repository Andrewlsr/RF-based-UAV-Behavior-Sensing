"""
Run lightweight release reproducibility checks.

This script checks imports, model artifact loading, optional prediction on one
recording, output folder organization, and hardcoded local path patterns in the
release-facing source files. It is intentionally lightweight so it can be run
before publishing or after cloning the repository.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy
import scipy
import sklearn

from src.model import load_model_artifacts
from src.prediction import predict_air_recording


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reproducibility_check.md"


def scan_for_absolute_paths() -> list[tuple[Path, int, str]]:
    """
    Scan release-facing files for Windows absolute path literals.

    Returns:
        Tuples of `(path, line_number, line_text)` where local drive paths were
        found.

    Notes:
        Generated outputs are not scanned here because they may contain local
        paths from a user-run demo; this check focuses on source and docs.
    """
    files = [
        *Path(PROJECT_ROOT / "src").glob("*.py"),
        *Path(PROJECT_ROOT / "scripts").glob("*.py"),
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / ".vscode" / "tasks.json",
    ]
    pattern = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\(?![nrt\"'])")
    matches = []
    for file_path in files:
        if not file_path.exists():
            continue
        for line_number, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if pattern.search(line):
                matches.append((file_path, line_number, line.strip()))
    return matches


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line check options.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Run release reproducibility checks.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    """
    Run checks and write a Markdown reproducibility report.

    Returns:
        None.

    Raises:
        SystemExit: If any check fails.
    """
    args = parse_arguments()
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "Python package imports",
            True,
            (
                f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
                f"matplotlib {matplotlib.__version__}, "
                f"scikit-learn {sklearn.__version__}, joblib {joblib.__version__}"
            ),
        )
    )

    try:
        classifier, scaler, config = load_model_artifacts(args.model_dir)
        checks.append(
            (
                "Model artifact loading",
                True,
                (
                    f"Loaded classifier {type(classifier).__name__}, scaler "
                    f"{type(scaler).__name__}, {len(config.get('feature_names', []))} features."
                ),
            )
        )
    except Exception as error:
        checks.append(("Model artifact loading", False, str(error)))

    if args.prediction_file is not None:
        try:
            result = predict_air_recording(args.prediction_file, args.model_dir)
            checks.append(
                (
                    "Prediction without retraining",
                    True,
                    (
                        f"{args.prediction_file.name}: {result.final_prediction} "
                        f"confidence {result.confidence * 100:.1f}%"
                    ),
                )
            )
        except Exception as error:
            checks.append(("Prediction without retraining", False, str(error)))
    else:
        checks.append(
            (
                "Prediction without retraining",
                True,
                "Skipped because --prediction-file was not supplied.",
            )
        )

    expected_paths = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "docs",
    ]
    missing = [path for path in expected_paths if not path.exists()]
    checks.append(
        (
            "Output path organization",
            not missing,
            "All expected release folders exist." if not missing else f"Missing: {missing}",
        )
    )

    path_matches = scan_for_absolute_paths()
    checks.append(
        (
            "No hardcoded local absolute paths in release-facing files",
            not path_matches,
            (
                "No Windows absolute path literals found."
                if not path_matches
                else "; ".join(
                    f"{path}:{line_number}" for path, line_number, _ in path_matches
                )
            ),
        )
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Reproducibility Check", ""]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- **{status}** `{name}`: {detail}")
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nReproducibility checks")
    failed = False
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failed = failed or not passed
        print(f"{status}: {name} - {detail}")
    print(f"Saved report: {args.report_path.resolve()}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
