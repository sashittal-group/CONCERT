from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MatplotlibPath
from PIL import Image

from .style import FONT_FAMILY, FONT_SIZE


FIGURE_SIZE = (3.02, 2.635)
PANEL_A_PDF_PATH = Path(__file__).with_name("tls_panelA.pdf")
PANEL_A_RENDER_WIDTH = 1800
FORMATS = ("png", "pdf", "svg")
STATE_COLORS = {
    "NMPs": "#8C8C8C",
    "NeuralTube": "#69B64F",
    "Somite": "#D393B1",
}

matplotlib.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})


def load_panel_a_image() -> Image.Image:
    if not PANEL_A_PDF_PATH.is_file():
        raise FileNotFoundError(f"Panel A source PDF not found: {PANEL_A_PDF_PATH}")

    with TemporaryDirectory(prefix="tls-panel-a-") as temporary_directory:
        image_path = Path(temporary_directory) / "panel_A.png"
        subprocess.run(
            [
                "sips",
                "-s",
                "format",
                "png",
                "--resampleWidth",
                str(PANEL_A_RENDER_WIDTH),
                str(PANEL_A_PDF_PATH),
                "--out",
                str(image_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with Image.open(image_path) as image:
            return image.convert("RGBA").copy()


def draw_panel_a(axis: Axes, panel_image: Image.Image) -> None:
    axis.imshow(panel_image, aspect="equal", interpolation="lanczos")
    axis.set_axis_off()


def draw_legacy_panel_a(axis: Axes, *, font_size: int = FONT_SIZE) -> None:
    axis.set_xlim(-0.74, 1.03)
    axis.set_ylim(0.03, 1.0)
    axis.set_aspect("auto")
    axis.axis("off")

    legend_rows = (
        ("NMPs", 0.60),
        ("Somite", 0.47),
        ("Neural Tube", 0.34),
    )
    for label, y_position in legend_rows:
        axis.text(
            -0.39,
            y_position,
            label,
            ha="right",
            va="center",
            fontsize=font_size,
        )
        axis.add_patch(
            Circle(
                (-0.31, y_position),
                radius=0.022,
                facecolor=STATE_COLORS[
                    "NeuralTube" if label == "Neural Tube" else label
                ],
                edgecolor="none",
                zorder=4,
            )
        )

    tissue = MatplotlibPath(
        [
            (0.03, 0.96),
            (0.03, 0.22),
            (0.03, 0.08),
            (0.18, 0.07),
            (0.52, 0.07),
            (0.87, 0.07),
            (0.97, 0.20),
            (0.97, 0.96),
        ],
        [
            MatplotlibPath.MOVETO,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.LINETO,
        ],
    )
    axis.add_patch(
        PathPatch(
            tissue,
            fill=False,
            edgecolor="#B8B8B8",
            linewidth=1.5,
            capstyle="round",
            joinstyle="round",
            zorder=0,
        )
    )
    axis.plot(
        (0.03, 0.97),
        (0.96, 0.96),
        linestyle=(0, (4, 3)),
        color="#B8B8B8",
        linewidth=1.5,
        zorder=0,
    )

    axis.add_patch(
        FancyBboxPatch(
            (0.43, 0.27),
            0.17,
            0.66,
            boxstyle="round,pad=0,rounding_size=0.015",
            facecolor=STATE_COLORS["NeuralTube"],
            edgecolor="none",
            alpha=0.58,
            zorder=1,
        )
    )

    for y_position in (0.77, 0.63, 0.49):
        for x_position in (0.12, 0.72):
            axis.add_patch(
                FancyBboxPatch(
                    (x_position, y_position),
                    0.17,
                    0.105,
                    boxstyle="round,pad=0,rounding_size=0.025",
                    facecolor=STATE_COLORS["Somite"],
                    edgecolor="none",
                    alpha=0.82,
                    zorder=2,
                )
            )
    for x_position in (0.12, 0.72):
        axis.add_patch(
            FancyBboxPatch(
                (x_position, 0.26),
                0.17,
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
            (0.515, 0.17),
            width=0.40,
            height=0.14,
            facecolor=STATE_COLORS["NMPs"],
            edgecolor="none",
            alpha=0.88,
            zorder=2,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.48, 0.20),
            (0.285, 0.49),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#111111",
            zorder=3,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.53, 0.21),
            (0.515, 0.49),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#111111",
            zorder=3,
        )
    )
    axis.add_patch(
        Circle(
            (0.285, 0.50),
            radius=0.028,
            facecolor="#A84479",
            edgecolor="none",
            zorder=4,
        )
    )
    axis.add_patch(
        Circle(
            (0.515, 0.50),
            radius=0.028,
            facecolor="#5FA64B",
            edgecolor="none",
            zorder=4,
        )
    )


def generate_panel_a(
    output_dir: Path,
    *,
    figsize: tuple[float, float] = FIGURE_SIZE,
    panel_image: Image.Image | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if panel_image is None:
        panel_image = load_panel_a_image()

    png_path = output_dir / "panel_A.png"
    panel_image.save(png_path)
    pdf_path = output_dir / "panel_A.pdf"
    shutil.copyfile(PANEL_A_PDF_PATH, pdf_path)

    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=figsize)
    draw_panel_a(axis, panel_image)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    svg_path = output_dir / "panel_A.svg"
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path, "svg": svg_path}


if __name__ == "__main__":
    from .generate import DEFAULT_OUTPUT_DIR

    for saved_path in generate_panel_a(DEFAULT_OUTPUT_DIR).values():
        print(f"Saved {saved_path}")
