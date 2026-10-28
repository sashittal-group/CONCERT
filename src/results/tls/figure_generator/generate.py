from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib.ticker import MaxNLocator
from PIL import Image

from src.summarize_fixed_sampling_transition_runs import RunParameters

from .cell_colors import CELL_TYPE_COLORS
from .panel_a import draw_panel_a, generate_panel_a, load_panel_a_image
from .panel_b import (
    COMPOSITE_FONT_SIZE as PANEL_B_COMPOSITE_FONT_SIZE,
    draw_panel_b as draw_convergence_panel_b,
    generate_panel_b as generate_convergence_panel_b,
    load_convergence_data,
)
from .panel_sensitivity import (
    PANEL_C_COMPOSITE_DATASET_FONT_SIZE,
    PANEL_C_COMPOSITE_FONT_SIZE,
    PANEL_C_WIDTH_RATIOS,
    ProfilePanelData,
    draw_boxplot_panel,
    draw_boxplot_legend,
    generate_boxplot_panel,
    load_profile_panel_data,
)
from .style import (
    FONT_FAMILY,
    FONT_SIZE,
    PANEL_BC_FONT_SIZE,
    PANEL_LABEL_SIZE,
)

matplotlib.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPOSITORY_ROOT / "data" / "troupe"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "figures" / "figure_generator"
STATES = ("NMPs", "Somite", "NeuralTube")
STATE_COLORS = CELL_TYPE_COLORS
NODE_POSITIONS = {
    "NMPs": (0.50, 0.78),
    "NeuralTube": (0.03, 0.10),
    "Somite": (0.97, 0.10),
}
FORMATS = ("png", "pdf", "svg")
CONDITION_SUBPANEL_FIGURE_SIZE = (3.8, 3.8)
COMPOSITE_FIGURE_SIZE = (16.5, 5.2)
PANEL_GRID_HEIGHT_RATIOS = (0.20, 0.80)
PANEL_GRID_HSPACE = 0.03


@dataclass(frozen=True)
class Fit:
    dataset: str
    source: Path
    completed: bool
    iteration: int | None
    log_likelihood: float
    birth: tuple[float, ...]
    death: tuple[float, ...]
    sampling: tuple[float, ...]
    transitions: dict[tuple[int, int], float]
    delta_t: float

    def net_growth(self, state_index: int) -> float:
        return self.birth[state_index] - self.death[state_index]

    def scaled_birth(self, state_index: int) -> float:
        return self.birth[state_index] * self.sampling[state_index]

    def transition_rate(self, target_state: int) -> float:
        if target_state == 0:
            probability = self.transitions.get((0, 0))
            if probability is None:
                probability = 1.0 - self.transitions[(0, 1)] - self.transitions[(0, 2)]
        else:
            probability = self.transitions[(0, target_state)]
        return probability / self.delta_t


def _read_vector(lines: list[str], name: str, source: Path) -> tuple[float, ...]:
    matches = [line for line in lines if line.startswith(f"{name}:")]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name!r} vector in {source}; found {len(matches)}")
    try:
        values = tuple(float(value) for value in matches[0].split(":", 1)[1].split())
    except ValueError as error:
        raise ValueError(f"Invalid {name!r} vector in {source}") from error
    if len(values) != len(STATES) or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected {len(STATES)} finite {name!r} values in {source}")
    return values


def _read_transition_spacing(path: Path) -> float:
    try:
        times = tuple(float(line.strip()) for line in path.read_text().splitlines() if line.strip())
    except ValueError as error:
        raise ValueError(f"Invalid transition time in {path}") from error
    if len(times) != 100 or not all(math.isfinite(value) for value in times):
        raise ValueError(f"Expected 100 finite transition times in {path}; found {len(times)}")
    differences = [right - left for left, right in zip(times, times[1:])]
    if not differences or any(value <= 0 for value in differences):
        raise ValueError(f"Transition times in {path} must be strictly increasing")
    delta_t = sorted(differences)[len(differences) // 2]
    if not math.isclose(delta_t, 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Expected nominal transition spacing 0.01 in {path}; found {delta_t}")
    return delta_t


def load_fit(dataset: str, data_root: Path = DATA_ROOT) -> Fit:
    if dataset not in {"TLS", "TLSC"}:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected 'TLS' or 'TLSC'")

    run_dir = data_root / dataset / "fixed_sampling_0.1"
    final_path = run_dir / "inference_transition_times_100.out"
    checkpoint_path = run_dir / "inference_transition_times_100.tmp.out"
    source = final_path if final_path.is_file() else checkpoint_path
    if not source.is_file():
        raise FileNotFoundError(f"No 100-transition fit or checkpoint found for {dataset}: {source}")

    lines = source.read_text().splitlines()
    header = next(
        (line for line in lines if re.fullmatch(r"(?:ITER\s+\d+\s+)?LL=\S+", line)),
        None,
    )
    if header is None:
        raise ValueError(f"Missing log-likelihood header in {source}")
    header_match = re.fullmatch(r"(?:ITER\s+(\d+)\s+)?LL=(\S+)", header)
    if header_match is None:
        raise ValueError(f"Invalid log-likelihood header in {source}: {header}")
    iteration = int(header_match.group(1)) if header_match.group(1) else None
    log_likelihood = float(header_match.group(2))
    if not math.isfinite(log_likelihood):
        raise ValueError(f"Invalid log likelihood in {source}")

    birth = _read_vector(lines, "birth", source)
    death = _read_vector(lines, "death", source)
    sampling = _read_vector(lines, "sampling", source)

    transitions: dict[tuple[int, int], float] = {}
    in_transition_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == "transitions:":
            in_transition_section = True
            continue
        if stripped == "transition_times:":
            break
        if in_transition_section:
            match = re.fullmatch(r"(\d+)->(\d+)\s+(\S+)", stripped)
            if match:
                source_state, target_state, probability = match.groups()
                value = float(probability)
                if not math.isfinite(value):
                    raise ValueError(f"Invalid transition probability in {source}: {stripped}")
                transitions[(int(source_state), int(target_state))] = value

    required_edges = {(0, 1), (0, 2)}
    if not required_edges.issubset(transitions):
        raise ValueError(f"Missing required NMP transition probabilities in {source}")

    delta_t = _read_transition_spacing(data_root / "transition-times-100.txt")
    return Fit(
        dataset=dataset,
        source=source,
        completed=final_path.is_file(),
        iteration=iteration,
        log_likelihood=log_likelihood,
        birth=birth,
        death=death,
        sampling=sampling,
        transitions=transitions,
        delta_t=delta_t,
    )


def _edge_width(rate: float, maximum_rate: float) -> float:
    minimum_width = 1.2
    maximum_width = 4.0
    relative_rate = math.log1p(rate) / math.log1p(maximum_rate)
    return minimum_width + (maximum_width - minimum_width) * relative_rate


def _arrow_style(rate: float, maximum_rate: float) -> str:
    relative_rate = math.log1p(rate) / math.log1p(maximum_rate)
    head_length = 0.45 + 0.45 * relative_rate
    head_width = 0.25 + 0.25 * relative_rate
    return (
        "-|>,"
        f"head_length={head_length:.3f},"
        f"head_width={head_width:.3f}"
    )


def _draw_edge(
    axis: Axes,
    source_position: tuple[float, float],
    target_position: tuple[float, float],
    label: str,
    source_radius: float,
    target_radius: float,
    transition_rate: float,
    line_width: float,
    maximum_transition_rate: float,
) -> None:
    dx = target_position[0] - source_position[0]
    dy = target_position[1] - source_position[1]
    distance = math.hypot(dx, dy)
    unit_x, unit_y = dx / distance, dy / distance
    start = (
        source_position[0] + source_radius * unit_x,
        source_position[1] + source_radius * unit_y,
    )
    end = (
        target_position[0] - target_radius * unit_x,
        target_position[1] - target_radius * unit_y,
    )
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=_arrow_style(transition_rate, maximum_transition_rate),
            mutation_scale=17,
            linewidth=line_width,
            color="#555555",
            joinstyle="miter",
            zorder=1,
        )
    )

    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    normal = (-unit_y, unit_x)
    side = -1 if target_position[0] < source_position[0] else 1
    label_position = (
        midpoint[0] + side * 0.12 * normal[0],
        midpoint[1] + side * 0.12 * normal[1],
    )
    axis.text(
        *label_position,
        label,
        ha="center",
        va="center",
        fontsize=PANEL_BC_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=3,
    )


def draw_condition(
    axis: Axes,
    fit: Fit,
    title: str,
    maximum_net_growth: float,
    maximum_transition_rate: float,
) -> None:
    axis.set_title(title, fontsize=PANEL_BC_FONT_SIZE, pad=2)
    axis.set_xlim(-0.20, 1.20)
    axis.set_ylim(-0.22, 1.52)
    axis.set_aspect("equal")
    axis.axis("off")

    state_indices = {"NMPs": 0, "Somite": 1, "NeuralTube": 2}
    minimum_radius = 0.12
    maximum_radius = 0.36
    radii = {}
    for state, index in state_indices.items():
        growth = max(0.0, fit.net_growth(index))
        growth_fraction = growth / maximum_net_growth if maximum_net_growth > 0 else 0.0
        radii[state] = math.sqrt(
            minimum_radius**2
            + growth_fraction * (maximum_radius**2 - minimum_radius**2)
        )

    for state, target_label, target_index in (
        ("NeuralTube", "NT", 2),
        ("Somite", "S", 1),
    ):
        _draw_edge(
            axis,
            NODE_POSITIONS["NMPs"],
            NODE_POSITIONS[state],
            rf"$p_{{\mathrm{{NMP,{target_label}}}}}/"
            rf"\Delta t={fit.transition_rate(target_index):.2f}$",
            source_radius=radii["NMPs"],
            target_radius=radii[state],
            transition_rate=fit.transition_rate(target_index),
            line_width=_edge_width(
                fit.transition_rate(target_index),
                maximum_transition_rate,
            ),
            maximum_transition_rate=maximum_transition_rate,
        )

    nmp_x, nmp_y = NODE_POSITIONS["NMPs"]
    nmp_radius = radii["NMPs"]
    loop_start_offset = 0.5 * nmp_radius
    loop_start_y = nmp_y + math.sqrt(nmp_radius**2 - loop_start_offset**2)
    loop_control_y = 1.56
    loop_path = MatplotlibPath(
        [
            (nmp_x - loop_start_offset, loop_start_y),
            (0.28, loop_control_y),
            (0.72, loop_control_y),
            (nmp_x + loop_start_offset, loop_start_y),
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
            arrowstyle=_arrow_style(
                fit.transition_rate(0),
                maximum_transition_rate,
            ),
            mutation_scale=17,
            linewidth=_edge_width(fit.transition_rate(0), maximum_transition_rate),
            color="#555555",
            joinstyle="miter",
            zorder=1,
        )
    )
    axis.text(
        nmp_x,
        1.27,
        rf"$p_{{\mathrm{{NMP,NMP}}}}/"
        rf"\Delta t={fit.transition_rate(0):.2f}$",
        ha="center",
        va="center",
        fontsize=PANEL_BC_FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.96,
        },
        zorder=3,
    )

    display_names = {
        "NMPs": "NMPs",
        "NeuralTube": "NT",
        "Somite": "S",
    }
    state_symbols = {
        "NMPs": r"\mathrm{NMP}",
        "NeuralTube": r"\mathrm{NT}",
        "Somite": r"\mathrm{S}",
    }
    for state, (x_position, y_position) in NODE_POSITIONS.items():
        index = state_indices[state]
        axis.add_patch(
            Circle(
                (x_position, y_position),
                radius=radii[state],
                facecolor=STATE_COLORS[state],
                edgecolor="white",
                linewidth=1.2,
                alpha=0.82,
                zorder=2,
            )
        )
        axis.text(
            x_position,
            y_position,
            f"{display_names[state]}\n"
            rf"$\beta_{{{state_symbols[state]}}} - "
            rf"\delta_{{{state_symbols[state]}}} "
            rf"= {fit.net_growth(index):.2f}$"
            "\n"
            rf"$\beta_{{{state_symbols[state]}}}\rho "
            rf"= {fit.scaled_birth(index):.2f}$",
            ha="center",
            va="center",
            fontsize=PANEL_BC_FONT_SIZE,
            linespacing=0.95,
            color="#202020",
            zorder=4,
        )


def _save_figure(figure: Figure, basename: Path) -> dict[str, Path]:
    basename.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for file_format in FORMATS:
        output_path = basename.with_suffix(f".{file_format}")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    return outputs


def _condition_title(fit: Fit) -> str:
    return "TLS" if fit.dataset == "TLS" else r"TLS$^{\mathrm{CL}}$"


def _figure_scales(*fits: Fit) -> tuple[float, float]:
    maximum_net_growth = max(
        0.0,
        *(fit.net_growth(index) for fit in fits for index in range(len(STATES))),
    )
    maximum_transition_rate = max(
        fit.transition_rate(target_state)
        for fit in fits
        for target_state in range(len(STATES))
    )
    return maximum_net_growth, maximum_transition_rate


def generate_condition_subpanel(
    fit: Fit,
    output_dir: Path,
    *,
    maximum_net_growth: float | None = None,
    maximum_transition_rate: float | None = None,
) -> dict[str, Path]:
    if maximum_net_growth is None or maximum_transition_rate is None:
        local_growth, local_rate = _figure_scales(fit)
        maximum_net_growth = (
            local_growth if maximum_net_growth is None else maximum_net_growth
        )
        maximum_transition_rate = (
            local_rate
            if maximum_transition_rate is None
            else maximum_transition_rate
        )
    figure, axis = plt.subplots(figsize=CONDITION_SUBPANEL_FIGURE_SIZE)
    draw_condition(
        axis,
        fit,
        _condition_title(fit),
        maximum_net_growth,
        maximum_transition_rate,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
    outputs = _save_figure(figure, output_dir / f"subpanel_{fit.dataset}")
    plt.close(figure)
    return outputs


def generate_panel_b(
    data_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    return generate_convergence_panel_b(data_root, output_dir)


def _match_y_axis_scale(axes: tuple[Axes, Axes]) -> None:
    lower = min(axis.get_ylim()[0] for axis in axes)
    upper = max(axis.get_ylim()[1] for axis in axes)
    ticks = MaxNLocator(nbins=5).tick_values(lower, upper)
    for axis in axes:
        axis.set_ylim(ticks[0], ticks[-1])
        axis.set_yticks(ticks)


def generate_composite(
    output_dir: Path,
    profile_panel_data: ProfilePanelData,
    runs_by_dataset: dict[str, list[RunParameters]],
    *,
    panel_a_image: Image.Image | None = None,
) -> dict[str, Path]:
    if panel_a_image is None:
        panel_a_image = load_panel_a_image()
    figure = plt.figure(figsize=COMPOSITE_FIGURE_SIZE)
    panel_grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(0.8, 2.0, 2.7),
        left=0.025,
        right=0.99,
        top=0.90,
        bottom=0.22,
        wspace=0.20,
    )

    panel_a_axis = figure.add_subplot(panel_grid[0, 0])
    draw_panel_a(panel_a_axis, panel_a_image)

    panel_b_grid = panel_grid[0, 1].subgridspec(
        2,
        3,
        height_ratios=PANEL_GRID_HEIGHT_RATIOS,
        hspace=PANEL_GRID_HSPACE,
        wspace=0.50,
    )
    panel_b_legend_axis = figure.add_subplot(panel_b_grid[0, :])
    panel_b_legend_axis.set_axis_off()
    panel_b_axes = tuple(
        figure.add_subplot(panel_b_grid[1, index]) for index in range(3)
    )
    draw_convergence_panel_b(
        panel_b_axes,
        runs_by_dataset,
        font_size=PANEL_B_COMPOSITE_FONT_SIZE,
        legend_axis=panel_b_legend_axis,
    )

    panel_c_grid = panel_grid[0, 2].subgridspec(
        2,
        3,
        height_ratios=PANEL_GRID_HEIGHT_RATIOS,
        width_ratios=PANEL_C_WIDTH_RATIOS,
        hspace=PANEL_GRID_HSPACE,
        wspace=0.43,
    )
    panel_c_legend_axis = figure.add_subplot(panel_c_grid[0, :])
    draw_boxplot_legend(
        panel_c_legend_axis,
        font_size=PANEL_C_COMPOSITE_FONT_SIZE,
    )
    panel_c_axes = tuple(
        figure.add_subplot(panel_c_grid[1, index]) for index in range(3)
    )
    draw_boxplot_panel(
        panel_c_axes,
        profile_panel_data,
        font_size=PANEL_C_COMPOSITE_FONT_SIZE,
        dataset_label_font_size=PANEL_C_COMPOSITE_DATASET_FONT_SIZE,
        group_label_y=-0.22,
    )
    _match_y_axis_scale((panel_b_axes[0], panel_c_axes[0]))
    _match_y_axis_scale((panel_b_axes[1], panel_c_axes[1]))

    for column, letter in enumerate(("A", "B", "C")):
        position = panel_grid[0, column].get_position(figure)
        figure.text(
            position.x0 + 0.006,
            position.y1 + 0.012,
            letter,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    outputs = _save_figure(figure, output_dir / "figure_composite")
    plt.close(figure)
    return outputs


def generate_figures(
    data_root: Path = DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, Path]]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    tls_fit = load_fit("TLS", data_root)
    tlsc_fit = load_fit("TLSC", data_root)
    maximum_net_growth, maximum_transition_rate = _figure_scales(tls_fit, tlsc_fit)
    profile_panel_data = load_profile_panel_data(data_root, output_dir)
    runs_by_dataset = load_convergence_data(data_root)
    panel_a_image = load_panel_a_image()

    outputs = {
        "panel_A": generate_panel_a(output_dir, panel_image=panel_a_image),
        "TLS": generate_condition_subpanel(
            tls_fit,
            output_dir,
            maximum_net_growth=maximum_net_growth,
            maximum_transition_rate=maximum_transition_rate,
        ),
        "TLSC": generate_condition_subpanel(
            tlsc_fit,
            output_dir,
            maximum_net_growth=maximum_net_growth,
            maximum_transition_rate=maximum_transition_rate,
        ),
        "panel_B": generate_panel_b(data_root, output_dir),
        "panel_C": generate_boxplot_panel(profile_panel_data, output_dir),
        "composite": generate_composite(
            output_dir,
            profile_panel_data,
            runs_by_dataset,
            panel_a_image=panel_a_image,
        ),
    }
    for panel_outputs in outputs.values():
        for path in panel_outputs.values():
            print(f"Saved {path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TLS/TLSC figure panels and composite."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing TLS/TLSC outputs and transition-time files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PNG, PDF, and SVG files.",
    )
    args = parser.parse_args()
    generate_figures(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
