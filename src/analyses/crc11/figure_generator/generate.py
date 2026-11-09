from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MatplotlibPath
import pandas as pd

from src.analyses.crc11.plot_state_specific_transitions import _read_result
from src.analyses.common.state_coloring_plot import _draw_subtree
from src.analyses.common.figure_style import FONT_FAMILY


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
COMBINED_INFERENCE_DIR = (
    REPOSITORY_ROOT / "data" / "sgootr" / "all_trees" / "combined_inference"
)
DEFAULT_FIT = (
    COMBINED_INFERENCE_DIR
    / "S10_CRC11_state_specific_3transition_shared_source_search"
    / "schedule_3857"
    / "inference.txt"
)
DEFAULT_FIGURES_DIR = (
    REPOSITORY_ROOT / "outputs" / "sgootr" / "CRC11" / "selected_fit" / "figures"
)
DEFAULT_TREE_SEGMENTS = DEFAULT_FIGURES_DIR / "CRC11_state_segments.csv"
DEFAULT_PROFILES_DIR = (
    REPOSITORY_ROOT
    / "outputs"
    / "sgootr"
    / "CRC11"
    / "selected_fit"
    / "profiles"
    / "data"
)
DEFAULT_COUNT_LIKELIHOOD = (
    REPOSITORY_ROOT
    / "outputs"
    / "sgootr"
    / "transition_count_analysis"
    / "S10_CRC11_ptln_transition_count_analysis"
    / "likelihood_by_ptln_transition_count.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs" / "sgootr" / "CRC11" / "manuscript_figure"
)

FORMATS = ("png", "pdf", "svg")
STATE_NAMES = ("NC", "PT", "LN")
STATE_COLORS = {
    "NC": "#6f91bd",
    "PT": "#e49a78",
    "LN": "#7b9e69",
}
GROWTH_PARAMETERS = tuple(f"growth[{state}]" for state in STATE_NAMES)
RELATIVE_BIRTH_PARAMETERS = ("birth_ratio[PT/NC]", "birth_ratio[LN/NC]")
PROFILE_BOX_LIKELIHOOD_FRACTION = math.exp(-0.25)
PROFILE_WHISKER_LIKELIHOOD_FRACTION = math.exp(-1.0)
FIGURE_FONT_SIZE = 16
PANEL_LABEL_SIZE = 18
PROFILE_XTICK_FONT_SIZE = 18

matplotlib.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": FIGURE_FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


@dataclass(frozen=True)
class ProfileInterval:
    parameter: str
    label: str
    estimate: float
    box_lower: float
    box_upper: float
    whisker_lower: float
    whisker_upper: float


@dataclass(frozen=True)
class FigureFit:
    log_likelihood: float
    birth: tuple[float, ...]
    death: tuple[float, ...]
    sampling: tuple[float, ...]
    transition_times: tuple[float, ...]
    epoch_edges: tuple[tuple[int, int], ...]
    transition_matrices: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def net_growth(self) -> tuple[float, ...]:
        return tuple(birth - death for birth, death in zip(self.birth, self.death))

    def event_probability(self, edge: tuple[int, int]) -> float:
        indices = [
            index for index, epoch_edge in enumerate(self.epoch_edges)
            if epoch_edge == edge
        ]
        if not indices:
            raise ValueError(f"The fitted result does not contain transition edge {edge}.")
        probabilities = [
            self.transition_matrices[index][edge[0]][edge[1]]
            for index in indices
        ]
        if any(
            not math.isclose(value, probabilities[0], rel_tol=0.0, abs_tol=1e-10)
            for value in probabilities[1:]
        ):
            raise ValueError(
                f"Repeated transition epochs for {edge} do not share a probability."
            )
        return probabilities[0]

    def self_probability(self, state: int, edge: tuple[int, int]) -> float:
        try:
            epoch_index = self.epoch_edges.index(edge)
        except ValueError as error:
            raise ValueError(f"The fitted result does not contain transition edge {edge}.") from error
        return self.transition_matrices[epoch_index][state][state]


def load_fit(path: Path) -> FigureFit:
    (
        likelihood,
        birth,
        death,
        sampling,
        transition_times,
        epoch_edges,
        transition_matrices,
    ) = _read_result(path)
    if len(transition_times) != 3 or epoch_edges != ((1, 2), (1, 2), (0, 1)):
        raise ValueError(
            "The manuscript figure expects two PT->LN epochs followed by "
            "one NC->PT epoch."
        )
    return FigureFit(
        log_likelihood=likelihood,
        birth=tuple(float(value) for value in birth.tolist()),
        death=tuple(float(value) for value in death.tolist()),
        sampling=tuple(float(value) for value in sampling.tolist()),
        transition_times=tuple(transition_times),
        epoch_edges=epoch_edges,
        transition_matrices=tuple(
            tuple(tuple(float(value) for value in row) for row in matrix.tolist())
            for matrix in transition_matrices
        ),
    )


def _read_parameter_profiles(
    fit: FigureFit,
    profiles_dir: Path,
) -> dict[str, list[ProfileInterval]]:
    metadata_path = profiles_dir / "parameter_metadata.csv"
    ranges_path = profiles_dir / "parameter_value_ranges.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Parameter metadata not found: {metadata_path}")
    if not ranges_path.is_file():
        raise FileNotFoundError(f"Parameter profile ranges not found: {ranges_path}")

    with metadata_path.open(newline="", encoding="utf-8") as source:
        metadata_rows = list(csv.DictReader(source))
    original_values = {
        row["parameter"]: float(row["original_value"])
        for row in metadata_rows
    }
    expected_values: dict[str, float] = {
        f"growth[{state}]": fit.net_growth[index]
        for index, state in enumerate(STATE_NAMES)
    }
    expected_values.update(
        {
            "birth_ratio[PT/NC]": fit.birth[1] / fit.birth[0],
            "birth_ratio[LN/NC]": fit.birth[2] / fit.birth[0],
        }
    )
    for parameter, expected_value in expected_values.items():
        estimate = original_values.get(parameter)
        if estimate is None or not math.isclose(
            estimate,
            expected_value,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"Profile metadata for {parameter} does not match the selected fit."
            )

    labels = {
        "growth[NC]": "NC",
        "growth[PT]": "PT",
        "growth[LN]": "LN",
        "birth_ratio[PT/NC]": r"$\beta_{\mathrm{PT}} / \beta_{\mathrm{NC}}$",
        "birth_ratio[LN/NC]": r"$\beta_{\mathrm{LN}} / \beta_{\mathrm{NC}}$",
    }
    parameter_types = {
        **dict.fromkeys(GROWTH_PARAMETERS, "growth"),
        **dict.fromkeys(RELATIVE_BIRTH_PARAMETERS, "birth_ratio"),
    }
    bounds: dict[str, dict[float, tuple[float, float]]] = {}
    with ranges_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            parameter = row["parameter"]
            if parameter not in parameter_types:
                continue
            if row["parameter_type"] != parameter_types[parameter]:
                raise ValueError(f"Unexpected profile type for {parameter}.")
            fraction = float(row["likelihood_fraction"])
            if fraction not in {
                PROFILE_BOX_LIKELIHOOD_FRACTION,
                PROFILE_WHISKER_LIKELIHOOD_FRACTION,
            }:
                continue
            if fraction in bounds.setdefault(parameter, {}):
                raise ValueError(f"Duplicate profile interval for {parameter}.")
            bounds[parameter][fraction] = (
                float(row["lower_parameter_value"]),
                float(row["higher_parameter_value"]),
            )

    intervals: dict[str, list[ProfileInterval]] = {
        "growth": [],
        "birth_ratio": [],
    }
    for parameter, parameter_type in parameter_types.items():
        if parameter not in original_values or set(bounds.get(parameter, {})) != {
            PROFILE_BOX_LIKELIHOOD_FRACTION,
            PROFILE_WHISKER_LIKELIHOOD_FRACTION,
        }:
            raise ValueError(f"Missing profile intervals for {parameter}.")
        box_lower, box_upper = bounds[parameter][PROFILE_BOX_LIKELIHOOD_FRACTION]
        whisker_lower, whisker_upper = bounds[parameter][
            PROFILE_WHISKER_LIKELIHOOD_FRACTION
        ]
        estimate = expected_values[parameter]
        if not (
            whisker_lower <= box_lower <= estimate <= box_upper <= whisker_upper
        ):
            raise ValueError(f"Inconsistent profile intervals for {parameter}.")
        intervals[parameter_type].append(
            ProfileInterval(
                parameter=parameter,
                label=labels[parameter],
                estimate=estimate,
                box_lower=box_lower,
                box_upper=box_upper,
                whisker_lower=whisker_lower,
                whisker_upper=whisker_upper,
            )
        )
    return intervals


def draw_panel_a(axis: Axes) -> None:
    axis.set_title(
        "Transition graph",
        loc="left",
        x=0.08,
        fontsize=FIGURE_FONT_SIZE,
        pad=5,
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.06, 1.08)
    axis.set_aspect("equal")
    axis.axis("off")

    positions = {"NC": (0.09, 0.43), "PT": (0.50, 0.43), "LN": (0.91, 0.43)}
    radius = 0.105
    for source_name, target_name in (("NC", "PT"), ("PT", "LN")):
        start_x = positions[source_name][0] + radius
        end_x = positions[target_name][0] - radius
        axis.add_patch(
            FancyArrowPatch(
                (start_x, positions[source_name][1]),
                (end_x, positions[target_name][1]),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#4d4d4d",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )
        midpoint = (start_x + end_x) / 2
        axis.text(
            midpoint,
            positions[source_name][1] + 0.13,
            f"{source_name}→{target_name}",
            ha="center",
            va="bottom",
            fontsize=FIGURE_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            zorder=3,
        )

    for state in positions:
        x_position, y_position = positions[state]
        loop_width = radius * 1.22
        loop_start_y = y_position + radius * 0.65
        loop_path = MatplotlibPath(
            [
                (x_position - loop_width * 0.5, loop_start_y),
                (x_position - loop_width * 0.72, y_position + 0.40),
                (x_position + loop_width * 0.72, y_position + 0.40),
                (x_position + loop_width * 0.5, loop_start_y),
            ],
            [
                MatplotlibPath.MOVETO,
                MatplotlibPath.CURVE4,
                MatplotlibPath.CURVE4,
                MatplotlibPath.CURVE4,
            ],
        )
        axis.add_patch(
            FancyArrowPatch(
                path=loop_path,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#4d4d4d",
                zorder=1,
            )
        )
    for state, (x_position, y_position) in positions.items():
        axis.add_patch(
            Circle(
                (x_position, y_position),
                radius=radius,
                facecolor=STATE_COLORS[state],
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )
        )
        axis.text(
            x_position,
            y_position,
            state,
            ha="center",
            va="center",
            fontsize=FIGURE_FONT_SIZE,
            fontweight="bold",
            color="#202020",
            zorder=3,
        )
def _draw_interval_profile(
    axis: Axes,
    intervals: list[ProfileInterval],
    colors_by_parameter: dict[str, str],
    *,
    ylabel: str,
) -> None:
    box_width = 0.15 * len(intervals)
    cap_half_width = box_width * 0.22
    for index, interval in enumerate(intervals):
        x = float(index)
        color = colors_by_parameter.get(interval.parameter, "#6f91bd")
        axis.vlines(
            x,
            interval.whisker_lower,
            interval.whisker_upper,
            color="#333333",
            linewidth=1.2,
            zorder=1,
        )
        axis.hlines(
            (interval.whisker_lower, interval.whisker_upper),
            x - cap_half_width,
            x + cap_half_width,
            color="#3F3F3F",
            linewidth=1.0,
            zorder=1,
        )
        axis.add_patch(
            Rectangle(
                (x - box_width / 2, interval.box_lower),
                box_width,
                interval.box_upper - interval.box_lower,
                facecolor=color,
                edgecolor="#3F3F3F",
                linewidth=0.8,
                alpha=0.62,
                zorder=2,
            )
        )
        axis.hlines(
            interval.estimate,
            x - box_width / 2,
            x + box_width / 2,
            color="#222222",
            linewidth=1.7,
            zorder=3,
        )
    axis.set_xticks(
        range(len(intervals)),
        labels=[interval.label for interval in intervals],
        rotation=0,
        ha="center",
    )
    axis.set_xlim(-0.5, len(intervals) - 0.5)
    axis.set_ylabel(ylabel, fontsize=FIGURE_FONT_SIZE)
    axis.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE, length=2, pad=2)
    axis.tick_params(
        axis="x",
        labelsize=PROFILE_XTICK_FONT_SIZE,
        length=2,
        pad=3,
    )
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_panel_d(
    growth_axis: Axes,
    birth_axis: Axes,
    profiles: dict[str, list[ProfileInterval]],
) -> None:
    growth_axis.set_title(
        "Net growth rate",
        fontsize=FIGURE_FONT_SIZE,
        pad=5,
    )
    _draw_interval_profile(
        growth_axis,
        profiles["growth"],
        {f"growth[{state}]": STATE_COLORS[state] for state in STATE_NAMES},
        ylabel=r"$\beta_s-\delta_s$",
    )
    birth_axis.set_title(
        "Relative birth rates",
        fontsize=FIGURE_FONT_SIZE,
        pad=5,
    )
    _draw_interval_profile(
        birth_axis,
        profiles["birth_ratio"],
        {
            "birth_ratio[PT/NC]": STATE_COLORS["PT"],
            "birth_ratio[LN/NC]": STATE_COLORS["LN"],
        },
        ylabel=r"$\beta_s/\beta_{\mathrm{NC}}$",
    )
    birth_axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)


def draw_panel_b(axis: Axes, count_likelihood_path: Path, fit: FigureFit) -> None:
    _draw_transition_count_likelihood(axis, count_likelihood_path, fit)


def draw_panel_c(
    tree_axis: Axes,
    tree_segments_path: Path,
    fit: FigureFit,
) -> float:
    if not tree_segments_path.is_file():
        raise FileNotFoundError(
            f"Colored CRC11 tree segments not found: {tree_segments_path}. "
            "Regenerate them with plot_state_specific_transitions."
        )
    tree_segments = pd.read_csv(tree_segments_path)
    if tree_segments.empty:
        raise ValueError(f"No colored tree segments in {tree_segments_path}.")
    root_time = float(
        max(
            tree_segments["segment_start_time"].max(),
            tree_segments["segment_end_time"].max(),
        )
    )
    if not math.isfinite(root_time) or root_time <= 0.0:
        raise ValueError(f"Invalid tree time range in {tree_segments_path}.")

    markers = tuple(
        (
            root_time - fitted_time,
            color,
        )
        for fitted_time, color in zip(
            fit.transition_times,
            ("#dd8452", "#c44e52", "#4c72b0"),
        )
    )
    drawn_root_time = _draw_subtree(
        tree_axis,
        tree_segments,
        "CRC11",
        "CRC11",
        STATE_NAMES,
        markers,
        STATE_COLORS,
        "CRC11 tree",
        FIGURE_FONT_SIZE,
    )
    if not math.isclose(drawn_root_time, root_time, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Tree-segment time range changed while drawing.")
    for line in tree_axis.lines:
        if line.get_linestyle() != "--":
            line.set_alpha(0.8)
    for collection in tree_axis.collections:
        collection.set_alpha(0.55)
    tree_ticks = [root_time * index / 4 for index in range(5)]
    tree_axis.set_yticks(
        [-value for value in tree_ticks],
        labels=[f"{value:.2f}" for value in tree_ticks],
    )
    tree_axis.set_ylabel("Time", fontsize=FIGURE_FONT_SIZE)
    tree_axis.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE)
    tree_legend = tree_axis.get_legend()
    if tree_legend is None:
        raise RuntimeError("The CRC11 tree state color map was not created.")
    tree_legend.set_title("")
    return root_time


def _draw_transition_count_likelihood(
    axis: Axes,
    path: Path,
    fit: FigureFit,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Transition-count likelihood data not found: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    counts: dict[int, tuple[float, int]] = {}
    for row in rows:
        count = int(row["num_ptln_transition_times"])
        ncpt_count = int(row["num_ncpt_transition_times"])
        likelihood = float(row["log_likelihood"])
        if count in counts or ncpt_count != 1 or not math.isfinite(likelihood):
            raise ValueError(f"Invalid transition-count likelihood row in {path}.")
        counts[count] = (likelihood, ncpt_count)
    if not counts or sorted(counts) != list(range(1, max(counts) + 1)):
        raise ValueError(f"Expected contiguous PT->LN counts in {path}.")
    selected_count = 2
    if selected_count not in counts or not math.isclose(
        counts[selected_count][0],
        fit.log_likelihood,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "The two-PT->LN likelihood does not match the selected fit."
        )

    counts_ordered = sorted(counts)
    selected_likelihood = counts[selected_count][0]
    deltas = [
        counts[count][0] - selected_likelihood
        for count in counts_ordered
    ]
    axis.plot(
        counts_ordered,
        deltas,
        color="#4c72b0",
        marker="o",
        linewidth=3.6,
        markersize=5,
        alpha=1.0,
        zorder=2,
    )
    axis.axhline(0.0, color="#555555", linewidth=0.8, zorder=1)
    axis.scatter(
        [selected_count],
        [0.0],
        marker="*",
        s=150,
        color="#e5a000",
        edgecolor="#333333",
        linewidth=0.4,
        zorder=4,
    )
    axis.set_xticks(counts_ordered)
    axis.set_xlabel(
        "Number of PT→LN transition times",
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_ylabel(
        r"$\Delta$ log-likelihood (vs. 2)",
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_title(
        "Log-likelihood change",
        fontsize=FIGURE_FONT_SIZE,
        pad=4,
    )
    axis.text(
        0.98,
        0.98,
        "* selected fit",
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_ylim(min(deltas) - 5.0, max(deltas) + 6.0)
    axis.tick_params(labelsize=FIGURE_FONT_SIZE)
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(figure: Figure, basename: Path) -> dict[str, Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for file_format in FORMATS:
        output_path = basename.with_suffix(f".{file_format}")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    return outputs


def _preserve_old_panel_outputs(output_dir: Path) -> None:
    for basename in ("panel_A", "panel_B", "panel_C", "figure_composite"):
        for file_format in FORMATS:
            current_path = output_dir / f"{basename}.{file_format}"
            old_path = output_dir / f"{basename}_old.{file_format}"
            if current_path.is_file() and not old_path.exists():
                shutil.copy2(current_path, old_path)


def _add_panel_label(
    figure: Figure,
    axis: Axes,
    letter: str,
    *,
    x_offset: float = -0.025,
    y_offset: float = 0.012,
) -> None:
    position = axis.get_position()
    figure.text(
        max(0.005, position.x0 + x_offset),
        position.y1 + y_offset,
        f"({letter.lower()})",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _write_profile_data(
    path: Path,
    profiles: dict[str, list[ProfileInterval]],
    fit: FigureFit,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "parameter",
                "estimate",
                "box_lower_likelihood_fraction_0.7788",
                "box_upper_likelihood_fraction_0.7788",
                "whisker_lower_likelihood_fraction_0.3679",
                "whisker_upper_likelihood_fraction_0.3679",
            ),
        )
        writer.writeheader()
        for interval in [*profiles["growth"], *profiles["birth_ratio"]]:
            writer.writerow(
                {
                    "parameter": interval.parameter,
                    "estimate": f"{interval.estimate:.15g}",
                    "box_lower_likelihood_fraction_0.7788": f"{interval.box_lower:.15g}",
                    "box_upper_likelihood_fraction_0.7788": f"{interval.box_upper:.15g}",
                    "whisker_lower_likelihood_fraction_0.3679": f"{interval.whisker_lower:.15g}",
                    "whisker_upper_likelihood_fraction_0.3679": f"{interval.whisker_upper:.15g}",
                }
            )
        for source, target in ((0, 1), (1, 2)):
            writer.writerow(
                {
                    "parameter": (
                        f"transition_probability[{STATE_NAMES[source]}->"
                        f"{STATE_NAMES[target]}]"
                    ),
                    "estimate": f"{fit.event_probability((source, target)):.15g}",
                }
            )
    return path


def generate_figures(
    inference_path: Path = DEFAULT_FIT,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    tree_segments: Path = DEFAULT_TREE_SEGMENTS,
    count_likelihood_path: Path = DEFAULT_COUNT_LIKELIHOOD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, Path]]:
    inference_path = inference_path.resolve()
    profiles_dir = profiles_dir.resolve()
    tree_segments = tree_segments.resolve()
    count_likelihood_path = count_likelihood_path.resolve()
    output_dir = output_dir.resolve()
    if not inference_path.is_file():
        raise FileNotFoundError(f"CRC11 inference fit not found: {inference_path}")
    fit = load_fit(inference_path)
    profiles = _read_parameter_profiles(fit, profiles_dir)
    if not tree_segments.is_file():
        raise FileNotFoundError(f"Colored CRC11 tree segments not found: {tree_segments}")
    if not count_likelihood_path.is_file():
        raise FileNotFoundError(
            f"Transition-count likelihood data not found: {count_likelihood_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _preserve_old_panel_outputs(output_dir)

    outputs: dict[str, dict[str, Path]] = {}
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    draw_panel_a(axis)
    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08)
    _add_panel_label(figure, axis, "A")
    outputs["panel_A"] = _save_figure(figure, output_dir / "panel_A")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 2.8))
    draw_panel_b(axis, count_likelihood_path, fit)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.24)
    _add_panel_label(figure, axis, "B")
    outputs["panel_B"] = _save_figure(figure, output_dir / "panel_B")
    plt.close(figure)

    figure, tree_axis = plt.subplots(figsize=(11.5, 3.3))
    draw_panel_c(tree_axis, tree_segments, fit)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.12)
    _add_panel_label(
        figure,
        tree_axis,
        "C",
        x_offset=-0.065,
        y_offset=0.075,
    )
    outputs["panel_C"] = _save_figure(figure, output_dir / "panel_C")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 3.7))
    panel_d_axes = tuple(axes)
    draw_panel_d(panel_d_axes[0], panel_d_axes[1], profiles)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.19, wspace=0.30)
    _add_panel_label(figure, panel_d_axes[0], "D")
    outputs["panel_D"] = _save_figure(figure, output_dir / "panel_D")
    plt.close(figure)

    figure = plt.figure(figsize=(18.5, 5.2))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.05, 1.75, 1.55),
        left=0.045,
        right=0.99,
        top=0.92,
        bottom=0.12,
        wspace=0.20,
    )
    left_grid = grid[0, 0].subgridspec(
        2,
        1,
        height_ratios=(1.15, 0.65),
        hspace=0.12,
    )
    panel_a_axis = figure.add_subplot(left_grid[0, 0])
    panel_b_axis = figure.add_subplot(left_grid[1, 0])
    panel_c_axis = figure.add_subplot(grid[0, 1])
    panel_d_grid = grid[0, 2].subgridspec(1, 2, wspace=0.30)
    panel_d_growth_axis = figure.add_subplot(panel_d_grid[0, 0])
    panel_d_birth_axis = figure.add_subplot(panel_d_grid[0, 1])

    draw_panel_a(panel_a_axis)
    draw_panel_b(panel_b_axis, count_likelihood_path, fit)
    draw_panel_c(panel_c_axis, tree_segments, fit)
    draw_panel_d(panel_d_growth_axis, panel_d_birth_axis, profiles)
    for axis, letter in (
        (panel_a_axis, "A"),
        (panel_b_axis, "B"),
        (panel_c_axis, "C"),
        (panel_d_growth_axis, "D"),
    ):
        _add_panel_label(
            figure,
            axis,
            letter,
            x_offset=-0.065 if letter == "C" else -0.025,
            y_offset=0.06 if letter == "C" else 0.012,
        )
    outputs["composite"] = _save_figure(figure, output_dir / "figure_composite")
    plt.close(figure)

    profile_csv = _write_profile_data(
        output_dir / "panel_D_profile_intervals.csv",
        profiles,
        fit,
    )
    outputs["profile_data"] = {"csv": profile_csv}
    for panel_outputs in outputs.values():
        for path in panel_outputs.values():
            print(f"Wrote {path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a four-panel SGOOTR CRC11 manuscript figure with "
            "transition graph, transition-count likelihood, colored tree, "
            "and parameter profile intervals."
        )
    )
    parser.add_argument("--inference", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--tree-segments",
        type=Path,
        default=DEFAULT_TREE_SEGMENTS,
    )
    parser.add_argument(
        "--count-likelihood",
        type=Path,
        default=DEFAULT_COUNT_LIKELIHOOD,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_figures(
        inference_path=args.inference,
        profiles_dir=args.profiles_dir,
        tree_segments=args.tree_segments,
        count_likelihood_path=args.count_likelihood,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
