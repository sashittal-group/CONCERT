from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib.text import Text

from .style import FONT_FAMILY, FONT_SIZE, PANEL_LABEL_SIZE


FORMATS = ("png", "pdf", "svg")
STATE_COLORS = {
    "NMPs": "#A6A6A6",
    "NeuralTube": "#69B64F",
    "Somite": "#D393B1",
}

matplotlib.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})


def draw_panel_a(axis: Axes) -> tuple[Text, ...]:
    axis.set_xlim(-0.22, 0.86)
    axis.set_ylim(0.02, 1.02)
    axis.set_aspect("auto")
    axis.axis("off")

    vessel_path = MatplotlibPath(
        [
            (0.08, 0.93),
            (0.08, 0.22),
            (0.08, 0.08),
            (0.80, 0.08),
            (0.80, 0.22),
            (0.80, 0.93),
        ],
        [
            MatplotlibPath.MOVETO,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.LINETO,
        ],
    )
    axis.add_patch(
        PathPatch(
            vessel_path,
            fill=False,
            edgecolor="#B8B8B8",
            linewidth=1.5,
            capstyle="round",
            joinstyle="round",
            zorder=0,
        )
    )
    axis.plot(
        (0.08, 0.80),
        (0.93, 0.93),
        linestyle=(0, (4, 3)),
        color="#B8B8B8",
        linewidth=1.5,
        zorder=0,
    )

    axis.add_patch(
        FancyBboxPatch(
            (0.36, 0.29),
            0.16,
            0.61,
            boxstyle="round,pad=0,rounding_size=0.015",
            facecolor=STATE_COLORS["NeuralTube"],
            edgecolor="none",
            alpha=0.58,
            zorder=1,
        )
    )

    somite_rows = (0.78, 0.64, 0.50)
    for y_position in somite_rows:
        for x_position in (0.17, 0.57):
            axis.add_patch(
                FancyBboxPatch(
                    (x_position, y_position),
                    0.15,
                    0.105,
                    boxstyle="round,pad=0,rounding_size=0.025",
                    facecolor=STATE_COLORS["Somite"],
                    edgecolor="none",
                    alpha=0.82,
                    zorder=2,
                )
            )
    for x_position in (0.17, 0.57):
        axis.add_patch(
            FancyBboxPatch(
                (x_position, 0.28),
                0.15,
                0.19,
                boxstyle="round,pad=0,rounding_size=0.035",
                facecolor=STATE_COLORS["Somite"],
                edgecolor="none",
                alpha=0.82,
                zorder=2,
            )
        )

    axis.add_patch(
        Ellipse(
            (0.44, 0.23),
            width=0.39,
            height=0.14,
            facecolor=STATE_COLORS["NMPs"],
            edgecolor="none",
            alpha=0.9,
            zorder=2,
        )
    )

    labels = (
        ("Neural\nTube", 0.63, 0.36),
        ("Somite", 0.43, 0.17),
        ("NMPs", 0.18, 0.325),
    )
    label_texts = []
    for label, y_position, target_x in labels:
        label_texts.append(
            axis.text(
                -0.17,
                y_position,
                label,
                ha="left",
                va="center",
                fontsize=FONT_SIZE,
                linespacing=1.15,
                color="#111111",
            )
        )

    return tuple(label_texts)


def draw_panel_a_arrows(axis: Axes, label_texts: tuple[Text, ...]) -> None:
    labels = (
        (0.63, 0.36),
        (0.43, 0.17),
        (0.18, 0.325),
    )
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    for (y_position, target_x), label_text in zip(
        labels,
        label_texts,
    ):
        label_right = label_text.get_window_extent(renderer).x1
        y_display = axis.transData.transform((0, y_position))[1]
        start_x = axis.transData.inverted().transform(
            (
                label_right + renderer.points_to_pixels(2),
                y_display,
            )
        )[0]
        minimum_arrow_length = 0.06
        if target_x - start_x < minimum_arrow_length:
            shift = minimum_arrow_length - (target_x - start_x)
            label_text.set_x(label_text.get_position()[0] - shift)
            axis.figure.canvas.draw()
            renderer = axis.figure.canvas.get_renderer()
            label_right = label_text.get_window_extent(renderer).x1
            start_x = axis.transData.inverted().transform(
                (
                    label_right + renderer.points_to_pixels(2),
                    y_display,
                )
            )[0]
        axis.add_patch(
            FancyArrowPatch(
                (start_x, y_position),
                (target_x, y_position),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.8,
                color="#111111",
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )


def generate_panel_a(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=(4.5, 4.3))
    label_texts = draw_panel_a(axis)
    figure.text(
        0.015,
        0.99,
        "A",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
    )
    figure.subplots_adjust(left=0.015, right=0.99, top=0.98, bottom=0.015)
    draw_panel_a_arrows(axis, label_texts)
    outputs = {}
    for file_format in FORMATS:
        output_path = output_dir / f"panel_A.{file_format}"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    from .generate import DEFAULT_OUTPUT_DIR

    for saved_path in generate_panel_a(DEFAULT_OUTPUT_DIR).values():
        print(f"Saved {saved_path}")
