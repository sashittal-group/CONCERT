from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PANEL_DEFINITIONS = {
    "D": ("growth", "panel_D"),
    "E": ("transition_ratio", "panel_E"),
    "F": ("scaled_sampling", "panel_F"),
}
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
            "dataset_labels": dict(zip(PROFILE_DATASET_IDS, DATASETS)),
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
    font_size: int = FONT_SIZE,
    dataset_label_font_size: int = FONT_SIZE,
) -> dict[str, Path]:
    try:
        _, basename = PANEL_DEFINITIONS[panel]
    except KeyError as error:
        raise ValueError(f"Unknown sensitivity panel {panel!r}") from error

    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
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
