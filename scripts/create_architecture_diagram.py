"""
Create the release architecture diagram.

The diagram communicates the project as a passive RF sensing framework:
UAV emission, SDR/IQ recording, signal processing, RF features, explainable
traditional ML, behaviour recognition, and airspace-monitoring support.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "system_architecture.png"


def add_box(axis, xy, width, height, text, facecolor, edgecolor="#2F3A45"):
    """
    Add a rounded process box to the architecture diagram.

    Args:
        axis: Matplotlib axis.
        xy: Lower-left box coordinate.
        width: Box width in axis coordinates.
        height: Box height in axis coordinates.
        text: Label shown inside the box.
        facecolor: Fill color.
        edgecolor: Outline color.
    """
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#17212B",
        weight="bold",
    )


def add_arrow(axis, start, end):
    """
    Add a pipeline arrow between diagram stages.

    Args:
        axis: Matplotlib axis.
        start: Arrow start coordinate.
        end: Arrow end coordinate.
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color="#2F3A45",
    )
    axis.add_patch(arrow)


def main() -> None:
    """
    Draw and save `docs/system_architecture.png`.

    Returns:
        None.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 12))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        "RF-based UAV Behaviour Sensing Framework",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#17212B",
    )
    axis.text(
        0.5,
        0.935,
        "Passive RF sensing first; machine learning is one component of the pipeline.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4D5B66",
    )

    boxes = [
        (0.25, 0.84, 0.50, 0.065, "UAV RF Emission", "#DCEBFF"),
        (0.25, 0.735, 0.50, 0.065, "SDR / IQ Recording", "#E6F4EA"),
        (0.20, 0.61, 0.60, 0.080, "IQ Signal Processing\nI/Q Reconstruction + DC Removal", "#FFF2CC"),
        (0.16, 0.485, 0.68, 0.080, "Frequency and Time-Frequency Analysis\nFFT | PSD | Spectrogram", "#FCE4D6"),
        (0.20, 0.36, 0.60, 0.080, "RF Feature Extraction\nPower | Entropy | Centroid | Activity", "#EADCF8"),
        (0.16, 0.235, 0.68, 0.080, "Explainable Traditional ML Model\nRandom Forest Flight-Mode Classifier", "#D9EAD3"),
        (0.20, 0.125, 0.60, 0.065, "UAV Behaviour Recognition\nON | HO | FY", "#DCEBFF"),
        (0.20, 0.035, 0.60, 0.060, "Airspace Monitoring Support", "#E6F4EA"),
    ]
    for x, y, width, height, text, color in boxes:
        add_box(axis, (x, y), width, height, text, color)

    arrow_points = [
        ((0.5, 0.84), (0.5, 0.80)),
        ((0.5, 0.735), (0.5, 0.69)),
        ((0.5, 0.61), (0.5, 0.565)),
        ((0.5, 0.485), (0.5, 0.44)),
        ((0.5, 0.36), (0.5, 0.315)),
        ((0.5, 0.235), (0.5, 0.19)),
        ((0.5, 0.125), (0.5, 0.095)),
    ]
    for start, end in arrow_points:
        add_arrow(axis, start, end)

    axis.text(
        0.5,
        0.005,
        "Output supports monitoring and risk assessment; it does not perform localization, tracking, or countermeasure decisions.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4D5B66",
    )

    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved architecture diagram: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
