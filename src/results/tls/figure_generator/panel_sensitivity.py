from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.results.tls.compare_transition_profiles import _read_profile_tables
from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.plot import PANELS, _plot_panel
from src.results.tls.figure_generator.cell_colors import CELL_TYPE_COLORS
from src.results.tls.figure_generator.style import (
    FONT_FAMILY,
    FONT_SIZE,
    PANEL_LABEL_SIZE,
)

matplotlib.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})


DATASETS = ("TLS", "TLSC")
PROFILE_DATASET_IDS = tuple(
    f"{dataset}_transition_times_100" for dataset in DATASETS
)
FORMATS = ("png", "pdf", "svg")
TRANSITION_RATIO_COLOR = "#A08870"
PANEL_FIGURE_SIZE = (5.2, 4.0)
PANEL_C_FIGURE_SIZE = (9.8, 4.0)
PANEL_C_FONT_SIZE = 12
PANEL_C_DATASET_FONT_SIZE = 12
PANEL_C_COMPOSITE_FONT_SIZE = 12
PANEL_C_COMPOSITE_DATASET_FONT_SIZE = 12
PANEL_DEFINITIONS = {
    "D": ("growth", "panel_D"),
    "E": ("transition_ratio", "panel_E"),
    "F": ("scaled_sampling", "panel_F"),
}
PANEL_C_COMPONENTS = ("D", "F", "E")
PANEL_C_WIDTH_RATIOS = (1.35, 1.35, 1.2)
PANEL_DETAILS = {
    parameter_type: (title, y_label)
    for title, parameter_type, y_label in PANELS
}
PANEL_Y_LABELS = {
    "E": r"$p_{\mathrm{NMP},\,\mathrm{S}} \,/\, p_{\mathrm{NMP},\,\mathrm{NT}}$",
    "F": r"$\beta_s \, \rho$",
}


@dataclass(frozen=True)
class ProfilePanelData:
    config: PipelineConfig
    ranges_by_dataset: dict[str, pd.DataFrame]
    metadata_by_dataset: dict[str, pd.DataFrame]


def load_profile_panel_data(
    data_root: Path,
    output_dir: Path,
) -> ProfilePanelData:
    config_path = Path(__file__).resolve().parents[1] / "config_fixed_sampling_0.1.yaml"
    base_config = load_config(config_path)
    ranges_by_dataset = {}
    metadata_by_dataset = {}
    parameter_sets = []
    likelihood_fractions = []

    for dataset_id, dataset in zip(PROFILE_DATASET_IDS, DATASETS):
        profile_dir = data_root / dataset / "fixed_sampling_0.1" / "profiles"
        ranges, metadata = _read_profile_tables(profile_dir, dataset)
        ranges_by_dataset[dataset_id] = ranges
        metadata_by_dataset[dataset_id] = metadata
        parameter_sets.append(set(metadata["parameter"]))
        likelihood_fractions.append(set(ranges["likelihood_fraction"]))

    if parameter_sets[0] != parameter_sets[1]:
        raise ValueError("TLS and TLSC profile tables contain different parameters.")
    if likelihood_fractions[0] != likelihood_fractions[1]:
        raise ValueError(
            "TLS and TLSC profile tables use different likelihood fractions."
        )

    plotting = dict(base_config.plotting)
    plotting.update(
        {
            "dataset_colors": {
                dataset_id: base_config.plotting["dataset_colors"][dataset]
                for dataset_id, dataset in zip(PROFILE_DATASET_IDS, DATASETS)
            },
            "dataset_labels": {
                PROFILE_DATASET_IDS[0]: "TLS",
                PROFILE_DATASET_IDS[1]: r"TLS$^{\mathrm{CL}}$",
            },
            "state_colors": dict(CELL_TYPE_COLORS),
            "formats": list(FORMATS),
        }
    )
    config = replace(
        base_config,
        datasets=PROFILE_DATASET_IDS,
        preprocessing_root=output_dir / "data",
        figure_root=output_dir,
        plotting=plotting,
    )
    return ProfilePanelData(
        config=config,
        ranges_by_dataset=ranges_by_dataset,
        metadata_by_dataset=metadata_by_dataset,
    )


def draw_profile_panel(
    axis: Axes,
    panel: str,
    data: ProfilePanelData,
    *,
    font_size: int,
    dataset_label_font_size: int,
) -> None:
    try:
        parameter_type, _ = PANEL_DEFINITIONS[panel]
        title, y_label = PANEL_DETAILS[parameter_type]
    except KeyError as error:
        raise ValueError(f"Unknown sensitivity panel {panel!r}") from error

    plotting = dict(data.config.plotting)
    plotting["font_size"] = font_size
    plotting["dataset_label_font_size"] = dataset_label_font_size
    if panel == "E":
        plotting["dataset_offset"] = 0.48
        plotting["dataset_colors"] = {
            dataset: TRANSITION_RATIO_COLOR
            for dataset in data.config.datasets
        }
    config = replace(data.config, plotting=plotting)
    y_label = PANEL_Y_LABELS.get(panel, y_label)
    _plot_panel(
        axis,
        title,
        parameter_type,
        y_label,
        data.ranges_by_dataset,
        data.metadata_by_dataset,
        config,
    )


def generate_profile_panel(
    panel: str,
    data: ProfilePanelData,
    output_dir: Path,
    *,
    figsize: tuple[float, float] = PANEL_FIGURE_SIZE,
    font_size: int = FONT_SIZE,
    dataset_label_font_size: int = FONT_SIZE,
) -> dict[str, Path]:
    try:
        _, basename = PANEL_DEFINITIONS[panel]
    except KeyError as error:
        raise ValueError(f"Unknown sensitivity panel {panel!r}") from error

    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=figsize)
    draw_profile_panel(
        axis,
        panel,
        data,
        font_size=font_size,
        dataset_label_font_size=dataset_label_font_size,
    )
    figure.text(
        0.005,
        0.995,
        panel,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
    )
    figure.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.16)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for file_format in FORMATS:
        output_path = output_dir / f"{basename}.{file_format}"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    plt.close(figure)
    return outputs


def draw_boxplot_panel(
    axes: tuple[Axes, Axes, Axes],
    data: ProfilePanelData,
    *,
    font_size: int = PANEL_C_FONT_SIZE,
    dataset_label_font_size: int = PANEL_C_DATASET_FONT_SIZE,
    group_label_y: float = -0.30,
) -> None:
    for axis, component in zip(axes, PANEL_C_COMPONENTS):
        draw_profile_panel(
            axis,
            component,
            data,
            font_size=font_size,
            dataset_label_font_size=dataset_label_font_size,
        )
        for tick in axis.get_xticklabels():
            tick.set_fontsize(dataset_label_font_size)
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
        for label in axis.texts:
            label.set_text(
                {
                    "NMPs": "NMP",
                    "NeuralTube": "NT",
                    "Somite / NeuralTube": "Somite / NT",
                }.get(label.get_text(), label.get_text())
            )
            label.set_fontsize(font_size)
            label.set_position((label.get_position()[0], group_label_y))

        axis.title.set_fontweight("medium")
        axis.yaxis.labelpad = 2
        axis.tick_params(axis="y", labelsize=font_size, length=2, pad=1)
        axis.tick_params(axis="x", length=2, pad=1)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.55, linestyle="-")
        for spine in ("left", "bottom"):
            axis.spines[spine].set_linewidth(0.7)
        for box_index, box in enumerate(axis.patches):
            dataset = data.config.datasets[
                box_index % len(data.config.datasets)
            ]
            dataset_label = data.config.plotting["dataset_labels"][dataset]
            state_color = box.get_facecolor()[:3]
            box.set_alpha(None)
            if dataset_label == "TLS":
                box.set_facecolor((*state_color, 0.62))
                box.set_edgecolor("#3F3F3F")
                box.set_linewidth(0.8)
            else:
                box.set_facecolor("white")
                box.set_edgecolor(state_color)
                box.set_linewidth(1.15)
        for line in axis.lines:
            line.set_linewidth(1.0 if line.get_zorder() >= 4 else 0.75)


def draw_boxplot_legend(
    axis: Axes,
    *,
    font_size: int,
    loc: str = "center",
) -> None:
    axis.set_axis_off()
    axis.legend(
        handles=(
            Patch(
                facecolor="#8C8C8C",
                edgecolor="#3F3F3F",
                alpha=0.62,
                label="TLS",
            ),
            Patch(
                facecolor="white",
                edgecolor="#777777",
                linewidth=1.15,
                label=r"TLS$^{\mathrm{CL}}$",
            ),
        ),
        loc=loc,
        ncols=2,
        frameon=False,
        fontsize=font_size,
        handlelength=1.1,
        handleheight=0.8,
        columnspacing=1.2,
    )


def generate_boxplot_panel(
    data: ProfilePanelData,
    output_dir: Path,
    *,
    figsize: tuple[float, float] = PANEL_C_FIGURE_SIZE,
    font_size: int = PANEL_C_FONT_SIZE,
    dataset_label_font_size: int = PANEL_C_DATASET_FONT_SIZE,
) -> dict[str, Path]:
    figure: Figure
    axes: tuple[Axes, Axes, Axes]
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(0.22, 0.78),
        width_ratios=PANEL_C_WIDTH_RATIOS,
        left=0.075,
        right=0.995,
        top=0.96,
        bottom=0.18,
        hspace=0.03,
        wspace=0.40,
    )
    legend_axis = figure.add_subplot(grid[0, :])
    draw_boxplot_legend(legend_axis, font_size=font_size, loc="upper center")
    axes = tuple(
        figure.add_subplot(grid[1, index])
        for index in range(len(PANEL_C_COMPONENTS))
    )
    draw_boxplot_panel(
        axes,
        data,
        font_size=font_size,
        dataset_label_font_size=dataset_label_font_size,
    )
    figure.text(
        0.005,
        0.995,
        "C",
        fontsize=PANEL_C_FONT_SIZE + 1,
        fontweight="bold",
        va="top",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for file_format in FORMATS:
        output_path = output_dir / f"panel_C.{file_format}"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    plt.close(figure)
    return outputs


def generate_panel_d(
    data: ProfilePanelData,
    output_dir: Path,
) -> dict[str, Path]:
    return generate_profile_panel("D", data, output_dir)


def generate_panel_e(
    data: ProfilePanelData,
    output_dir: Path,
) -> dict[str, Path]:
    return generate_profile_panel("E", data, output_dir)


def generate_panel_f(
    data: ProfilePanelData,
    output_dir: Path,
) -> dict[str, Path]:
    return generate_profile_panel("F", data, output_dir)
