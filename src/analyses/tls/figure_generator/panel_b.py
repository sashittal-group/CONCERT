from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.analyses.common.cell_state_colors import CELL_STATE_ORDER
from src.analyses.tls.inference_summaries import RunParameters, load_runs

from .cell_colors import CELL_TYPE_COLORS
from src.analyses.common.figure_style import (
    FONT_FAMILY,
    FONT_SIZE,
    PANEL_LABEL_SIZE,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPOSITORY_ROOT / "data" / "troupe"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "tls" / "figure_generator"
FIGURE_SIZE = (11.5, 4.5)
PANEL_FONT_SIZE = 12
COMPOSITE_FONT_SIZE = 12
MAX_TRANSITIONS = 100
FORMATS = ("png", "pdf", "svg")
DATASETS = ("TLS", "TLSC")
DATASET_STYLES = {
    "TLS": ("-", "o"),
    "TLSC": ("--", "s"),
}
STATE_COLORS = CELL_TYPE_COLORS

matplotlib.rcParams.update({"font.family": FONT_FAMILY, "font.size": FONT_SIZE})


def load_convergence_data(
    data_root: Path = DATA_ROOT,
) -> dict[str, list[RunParameters]]:
    data_by_dataset = {}
    for dataset in DATASETS:
        runs = load_runs(
            data_root / dataset / "fixed_sampling_0.1",
            max_count=MAX_TRANSITIONS,
        )
        selected = [run for run in runs if run.count <= MAX_TRANSITIONS]
        if not selected:
            raise ValueError(f"No {dataset} convergence runs at or below 100 transitions.")
        data_by_dataset[dataset] = selected
    return data_by_dataset


def draw_panel_b(
    axes: tuple[Axes, Axes, Axes],
    runs_by_dataset: dict[str, list[RunParameters]],
    *,
    panel_letter: bool = False,
    font_size: int = COMPOSITE_FONT_SIZE,
    legend_axis: Axes | None = None,
) -> None:
    if set(runs_by_dataset) != set(DATASETS):
        raise ValueError(f"Expected convergence runs for {DATASETS}.")
    metrics = (
        (
            "Net growth rate",
            r"$\beta_s-\delta_s$",
            lambda run, state: run.birth_rates[state] - run.death_rates[state],
            tuple(range(len(CELL_STATE_ORDER))),
        ),
        (
            "Scaled birth rate",
            r"$\beta_s \rho$",
            lambda run, state: run.birth_rates[state] * run.sampling_rates[state],
            tuple(range(len(CELL_STATE_ORDER))),
        ),
        (
            r"Transition probability / $\Delta t$",
            r"$p_{s,d} \,/\, \Delta t$",
            lambda run, target: run.transition_probabilities.get((0, target), 0.0)
            * run.count,
            (1, 2),
        ),
    )
    for axis, (title, ylabel, value_for_run, series_indices) in zip(axes, metrics):
        for dataset in DATASETS:
            linestyle, marker = DATASET_STYLES[dataset]
            for series_index in series_indices:
                state_name = (
                    CELL_STATE_ORDER[series_index]
                    if len(series_indices) > 2
                    else f"NMPs→{CELL_STATE_ORDER[series_index]}"
                )
                runs = runs_by_dataset[dataset]
                axis.plot(
                    [run.count for run in runs],
                    [value_for_run(run, series_index) for run in runs],
                    color=STATE_COLORS[state_name.split("→")[-1]],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.1,
                    label=f"{dataset} · {state_name}",
                )
        axis.set_title(title, fontsize=font_size, pad=5)
        axis.set_xlabel("Number of transitions", fontsize=font_size)
        axis.set_ylabel(ylabel, fontsize=font_size)
        axis.set_xlim(0, MAX_TRANSITIONS + 3)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.tick_params(labelsize=font_size)
        axis.grid(True, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    state_handles = [
        Line2D(
            [0],
            [0],
            color=STATE_COLORS[state],
            linewidth=2,
            label=state,
        )
        for state in CELL_STATE_ORDER
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=DATASET_STYLES[dataset][0],
            marker=DATASET_STYLES[dataset][1],
            label=dataset if dataset == "TLS" else r"TLS$^{\mathrm{CL}}$",
        )
        for dataset in DATASETS
    ]
    if legend_axis is not None:
        legend_axis.set_axis_off()
        legend_axis.legend(
            handles=state_handles + dataset_handles,
            ncols=5,
            fontsize=font_size,
            frameon=False,
            loc="center",
            handlelength=1.5,
            columnspacing=1.0,
        )
    else:
        axes[0].figure.legend(
            handles=state_handles + dataset_handles,
            ncols=5,
            fontsize=font_size,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            handlelength=1.5,
            columnspacing=1.0,
        )
    if panel_letter:
        axes[0].figure.text(
            0.005,
            0.995,
            "B",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
        )


def generate_panel_b(
    data_root: Path = DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    figsize: tuple[float, float] = FIGURE_SIZE,
) -> dict[str, Path]:
    runs_by_dataset = load_convergence_data(data_root)
    figure: Figure
    axes: tuple[Axes, Axes, Axes]
    figure, created_axes = plt.subplots(1, 3, figsize=figsize)
    axes = tuple(created_axes)
    draw_panel_b(
        axes,
        runs_by_dataset,
        panel_letter=True,
        font_size=PANEL_FONT_SIZE,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.90), w_pad=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for file_format in FORMATS:
        output_path = output_dir / f"panel_B.{file_format}"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs[file_format] = output_path
    plt.close(figure)
    return outputs


if __name__ == "__main__":
    for saved_path in generate_panel_b().values():
        print(f"Saved {saved_path}")
