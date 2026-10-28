from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import pandas as pd

from src.results.sgootr.plot_state_specific_transitions import _read_result
from src.results.tls.plot_state_coloring import _draw_subtree
from src.results.tls.figure_generator.style import (
    FONT_FAMILY,
)


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
DEFAULT_FIGURES_DIR = DEFAULT_FIT.parent / "figures"
DEFAULT_TREE_SEGMENTS = DEFAULT_FIGURES_DIR / "CRC11_state_segments.csv"
DEFAULT_PROFILES_DIR = DEFAULT_FIT.parent / "profiles" / "data"
DEFAULT_INPUT_DIR = COMBINED_INFERENCE_DIR / "S10_CRC11_transition_chain_input"
DEFAULT_TRANSITION_PROFILE = (
    COMBINED_INFERENCE_DIR
    / "S10_CRC11_ptln_transition_count_analysis"
    / "ptln_2"
    / "fixed_parameter_time_profile"
    / "fixed_parameter_transition_time_profile.csv"
)
DEFAULT_COUNT_LIKELIHOOD = (
    COMBINED_INFERENCE_DIR
    / "S10_CRC11_ptln_transition_count_analysis"
    / "likelihood_by_ptln_transition_count.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "sgootr" / "figures" / "figure_generator"

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
FIGURE_FONT_SIZE = 8

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
        "growth[NC]": r"$\gamma_{\mathrm{NC}}$",
        "growth[PT]": r"$\gamma_{\mathrm{PT}}$",
        "growth[LN]": r"$\gamma_{\mathrm{LN}}$",
        "birth_ratio[PT/NC]": r"$\beta_{\mathrm{PT}}/\beta_{\mathrm{NC}}$",
        "birth_ratio[LN/NC]": r"$\beta_{\mathrm{LN}}/\beta_{\mathrm{NC}}$",
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

    positions = {"NC": (0.16, 0.43), "PT": (0.50, 0.43), "LN": (0.84, 0.43)}
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
            x - 0.10,
            x + 0.10,
            color="#333333",
            linewidth=1.0,
            zorder=1,
        )
        axis.add_patch(
            Rectangle(
                (x - 0.23, interval.box_lower),
                0.46,
                interval.box_upper - interval.box_lower,
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.8,
                alpha=0.55,
                zorder=2,
            )
        )
        axis.hlines(
            interval.estimate,
            x - 0.23,
            x + 0.23,
            color="#222222",
            linewidth=1.7,
            zorder=3,
        )
    axis.set_xticks(
        range(len(intervals)),
        labels=[interval.label for interval in intervals],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylabel(ylabel, fontsize=FIGURE_FONT_SIZE)
    axis.tick_params(labelsize=FIGURE_FONT_SIZE, length=3)
    axis.grid(axis="y", linestyle=":", alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _transition_probability_matrix(
    fit: FigureFit,
    edge: tuple[int, int],
) -> np.ndarray:
    matrices = [
        np.asarray(matrix, dtype=float)
        for epoch_edge, matrix in zip(fit.epoch_edges, fit.transition_matrices)
        if epoch_edge == edge
    ]
    if not matrices:
        raise ValueError(f"The fitted result does not contain transition edge {edge}.")
    if any(not np.allclose(matrix, matrices[0], rtol=0.0, atol=1e-10) for matrix in matrices[1:]):
        raise ValueError(f"Repeated epochs for {edge} do not share a transition matrix.")
    return matrices[0]


def draw_panel_b(
    growth_axis: Axes,
    matrix_axes: tuple[Axes, Axes],
    birth_axis: Axes,
    profiles: dict[str, list[ProfileInterval]],
    fit: FigureFit,
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

    matrix_edges = ((0, 1), (1, 2))
    matrix_titles = ("NC→PT epoch", "PT→LN epochs (shared matrix)")
    for axis, edge, title in zip(matrix_axes, matrix_edges, matrix_titles):
        axis.set_title(title, fontsize=FIGURE_FONT_SIZE, pad=5)
        matrix = _transition_probability_matrix(fit, edge)
        axis.imshow(
            np.ma.masked_equal(matrix, 0.0),
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        for row in range(len(STATE_NAMES)):
            for column in range(len(STATE_NAMES)):
                probability = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{probability:.3f}" if probability else "0",
                    ha="center",
                    va="center",
                    fontsize=FIGURE_FONT_SIZE,
                    color="white" if probability >= 0.5 else "#222222",
                )
        axis.set_xticks(range(len(STATE_NAMES)), labels=STATE_NAMES)
        axis.set_yticks(range(len(STATE_NAMES)), labels=STATE_NAMES)
        axis.set_xlabel(
            "Destination",
            fontsize=FIGURE_FONT_SIZE,
            labelpad=2,
        )
        axis.tick_params(labelsize=FIGURE_FONT_SIZE, length=0)
        axis.set_aspect("equal", adjustable="box")
    matrix_axes[0].set_ylabel(
        "Source",
        fontsize=FIGURE_FONT_SIZE,
        labelpad=3,
    )
    matrix_axes[1].set_ylabel("")
    matrix_axes[1].tick_params(labelleft=False)

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


def draw_panel_c(
    tree_axis: Axes,
    likelihood_axis: Axes,
    count_axis: Axes,
    tree_segments_path: Path,
    transition_profile_path: Path,
    count_likelihood_path: Path,
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

    if not transition_profile_path.is_file():
        raise FileNotFoundError(
            f"Transition-time likelihood profile not found: {transition_profile_path}"
        )
    with transition_profile_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No transition-time profile points in {transition_profile_path}")
    for epoch_index, fitted_time in enumerate(fit.transition_times):
        baseline_rows = [
            row for row in rows
            if int(row["epoch_index"]) == epoch_index
            and math.isclose(
                float(row["varied_time"]),
                fitted_time,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ]
        if len(baseline_rows) != 1 or not math.isclose(
            float(baseline_rows[0]["log_likelihood"]),
            fit.log_likelihood,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "Transition-time profile does not match the selected fit at "
                f"epoch {epoch_index}."
            )

    styles = tuple(
        (fit.epoch_edges[index], index, color)
        for index, color in enumerate(("#dd8452", "#c44e52", "#4c72b0"))
    )
    finite_deltas = []
    series = []
    markers = []
    for edge, occurrence, color in styles:
        edge_label = f"{STATE_NAMES[edge[0]]}->{STATE_NAMES[edge[1]]}"
        label = (
            f"{edge_label.replace('->', '→')} "
            f"({fit.transition_times[occurrence]:.3f})"
        )
        markers.append((root_time - fit.transition_times[occurrence], color))
        edge_rows = [
            row for row in rows
            if row["transition_edge"] == edge_label
            and int(row["epoch_index"]) == occurrence
        ]
        if not edge_rows:
            raise ValueError(f"Missing transition-time profile for {label}.")
        points = [
            (float(row["varied_time"]), float(row["log_likelihood"]))
            for row in edge_rows
        ]
        deltas = [
            likelihood - fit.log_likelihood
            for _, likelihood in points
        ]
        finite_deltas.extend(value for value in deltas if math.isfinite(value))
        series.append((points, deltas, color, label, fit.transition_times[occurrence]))

    if not finite_deltas:
        raise FloatingPointError("No finite transition-time profile likelihoods.")
    drawn_root_time = _draw_subtree(
        tree_axis,
        tree_segments,
        "CRC11 S10",
        "CRC11",
        STATE_NAMES,
        tuple(markers),
        STATE_COLORS,
        "CRC11 S10 colored tree",
        FIGURE_FONT_SIZE,
    )
    if not math.isclose(drawn_root_time, root_time, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Tree-segment time range changed while drawing.")
    tree_axis.set_ylabel("Time from root", fontsize=FIGURE_FONT_SIZE)
    likelihood_axis.sharey(tree_axis)
    likelihood_axis.set_ylim(-root_time, 0.0)
    likelihood_floor = min(-5.0, min(finite_deltas) - 2.0)
    for points, deltas, color, label, baseline_time in series:
        time_values = np.asarray([time for time, _ in points], dtype=float)
        likelihood_values = np.asarray(
            [value if math.isfinite(value) else np.nan for value in deltas],
            dtype=float,
        )
        mirrored_times = root_time - time_values
        tree_aligned_times = -mirrored_times
        likelihood_axis.plot(
            likelihood_values,
            tree_aligned_times,
            color=color,
            linewidth=1.2,
            label=label,
            zorder=3,
        )
        fill_order = np.argsort(tree_aligned_times)
        likelihood_axis.fill_betweenx(
            tree_aligned_times[fill_order],
            likelihood_floor,
            likelihood_values[fill_order],
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        zero_likelihood_points = [
            time for (time, _), delta in zip(points, deltas)
            if delta == -math.inf
        ]
        if zero_likelihood_points:
            likelihood_axis.scatter(
                [likelihood_floor] * len(zero_likelihood_points),
                [-time for time in zero_likelihood_points],
                marker="v",
                color=color,
                s=FIGURE_FONT_SIZE * 2,
                zorder=3,
            )
        likelihood_axis.axhline(
            -(root_time - baseline_time),
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
        )
    likelihood_axis.axvline(0.0, color="#555555", linewidth=0.8)
    likelihood_axis.set_xlim(likelihood_floor, 1.0)
    likelihood_axis.set_xlabel(
        r"$\Delta$ log-likelihood",
        fontsize=FIGURE_FONT_SIZE,
    )
    likelihood_axis.set_ylabel("")
    likelihood_axis.set_title(
        "Likelihood by transition time",
        loc="left",
        fontsize=FIGURE_FONT_SIZE,
        pad=4,
    )
    likelihood_axis.tick_params(
        labelsize=FIGURE_FONT_SIZE,
        labelleft=False,
    )
    likelihood_axis.grid(True, axis="x", linestyle=":", alpha=0.4)
    likelihood_axis.legend(
        frameon=False,
        fontsize=FIGURE_FONT_SIZE,
        loc="lower right",
    )
    _draw_transition_count_likelihood(count_axis, count_likelihood_path, fit)
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
        linewidth=1.2,
        markersize=4,
        zorder=2,
    )
    axis.axhline(0.0, color="#555555", linewidth=0.8, zorder=1)
    axis.scatter(
        [selected_count],
        [0.0],
        marker="*",
        s=90,
        color="#e5a000",
        edgecolor="#333333",
        linewidth=0.4,
        zorder=4,
    )
    axis.set_xticks(counts_ordered)
    axis.set_xlabel("PT→LN count", fontsize=FIGURE_FONT_SIZE)
    axis.set_ylabel(
        r"$\Delta$ LL (vs 2)",
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_title(
        "PT→LN count",
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
    input_dir: Path = DEFAULT_INPUT_DIR,
    transition_profile_path: Path = DEFAULT_TRANSITION_PROFILE,
    count_likelihood_path: Path = DEFAULT_COUNT_LIKELIHOOD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, Path]]:
    inference_path = inference_path.resolve()
    profiles_dir = profiles_dir.resolve()
    tree_segments = tree_segments.resolve()
    input_dir = input_dir.resolve()
    transition_profile_path = transition_profile_path.resolve()
    count_likelihood_path = count_likelihood_path.resolve()
    output_dir = output_dir.resolve()
    if not inference_path.is_file():
        raise FileNotFoundError(f"CRC11 inference fit not found: {inference_path}")
    fit = load_fit(inference_path)
    profiles = _read_parameter_profiles(fit, profiles_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"CRC11 tree input directory not found: {input_dir}")
    if not transition_profile_path.is_file():
        raise FileNotFoundError(
            f"Transition-time likelihood profile not found: {transition_profile_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, Path]] = {}
    figure, axis = plt.subplots(figsize=(6.3, 4.2))
    draw_panel_a(axis)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06)
    figure.text(
        0.01,
        0.99,
        "A",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        va="top",
    )
    outputs["panel_A"] = _save_figure(figure, output_dir / "panel_A")
    plt.close(figure)

    figure = plt.figure(figsize=(15.5, 5.6))
    panel_b_grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.15, 0.9, 1.15),
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.12,
        wspace=0.22,
    )
    panel_b_growth_axis = figure.add_subplot(panel_b_grid[0, 0])
    panel_b_matrix_grid = panel_b_grid[0, 1].subgridspec(2, 1, hspace=0.30)
    panel_b_matrix_axes = tuple(
        figure.add_subplot(panel_b_matrix_grid[index, 0]) for index in range(2)
    )
    panel_b_birth_axis = figure.add_subplot(panel_b_grid[0, 2])
    draw_panel_b(
        panel_b_growth_axis,
        panel_b_matrix_axes,
        panel_b_birth_axis,
        profiles,
        fit,
    )
    figure.text(
        0.01,
        0.99,
        "B",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        va="top",
    )
    outputs["panel_B"] = _save_figure(figure, output_dir / "panel_B")
    plt.close(figure)

    figure = plt.figure(figsize=(14.5, 5.2))
    panel_c_grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.65, 0.36, 0.75),
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.13,
        wspace=0.18,
    )
    tree_axis = figure.add_subplot(panel_c_grid[0, 0])
    likelihood_axis = figure.add_subplot(panel_c_grid[0, 1])
    count_axis = figure.add_subplot(panel_c_grid[0, 2])
    draw_panel_c(
        tree_axis,
        likelihood_axis,
        count_axis,
        tree_segments,
        transition_profile_path,
        count_likelihood_path,
        fit,
    )
    figure.text(
        0.01,
        0.99,
        "C",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        va="top",
    )
    outputs["panel_C"] = _save_figure(figure, output_dir / "panel_C")
    plt.close(figure)

    figure = plt.figure(figsize=(16.0, 11.0))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.15, 0.85),
        left=0.04,
        right=0.985,
        top=0.95,
        bottom=0.08,
        hspace=0.20,
    )
    top_grid = grid[0, 0].subgridspec(
        1,
        2,
        width_ratios=(0.72, 2.40),
        wspace=0.10,
    )
    panel_a_axis = figure.add_subplot(top_grid[0, 0])
    panel_c_grid = top_grid[0, 1].subgridspec(
        1,
        3,
        width_ratios=(1.65, 0.36, 0.75),
        wspace=0.10,
    )
    panel_c_tree_axis = figure.add_subplot(panel_c_grid[0, 0])
    panel_c_likelihood_axis = figure.add_subplot(panel_c_grid[0, 1])
    panel_c_count_axis = figure.add_subplot(panel_c_grid[0, 2])
    panel_b_grid = grid[1, 0].subgridspec(
        1,
        3,
        width_ratios=(1.15, 0.9, 1.15),
        wspace=0.22,
    )
    panel_b_growth_axis = figure.add_subplot(panel_b_grid[0, 0])
    panel_b_matrix_grid = panel_b_grid[0, 1].subgridspec(2, 1, hspace=0.28)
    panel_b_matrix_axes = tuple(
        figure.add_subplot(panel_b_matrix_grid[index, 0]) for index in range(2)
    )
    panel_b_birth_axis = figure.add_subplot(panel_b_grid[0, 2])
    draw_panel_a(panel_a_axis)
    draw_panel_b(
        panel_b_growth_axis,
        panel_b_matrix_axes,
        panel_b_birth_axis,
        profiles,
        fit,
    )
    draw_panel_c(
        panel_c_tree_axis,
        panel_c_likelihood_axis,
        panel_c_count_axis,
        tree_segments,
        transition_profile_path,
        count_likelihood_path,
        fit,
    )
    for axis, letter in (
        (panel_a_axis, "A"),
        (panel_c_tree_axis, "C"),
        (panel_b_growth_axis, "B"),
    ):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.025,
            position.y1 + 0.012,
            letter,
            fontsize=FIGURE_FONT_SIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )
    outputs["composite"] = _save_figure(figure, output_dir / "figure_composite")
    plt.close(figure)

    profile_csv = _write_profile_data(
        output_dir / "panel_B_profile_intervals.csv",
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
            "Generate a three-panel SGOOTR CRC11 manuscript figure with "
            "transition graph, identifiable parameter estimates, and "
            "posterior-MAP colored tree."
        )
    )
    parser.add_argument("--inference", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--tree-segments",
        type=Path,
        default=DEFAULT_TREE_SEGMENTS,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--transition-profile",
        type=Path,
        default=DEFAULT_TRANSITION_PROFILE,
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
        input_dir=args.input_dir,
        transition_profile_path=args.transition_profile,
        count_likelihood_path=args.count_likelihood,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
