from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.analyses.crc11.preprocessing.transitions import STATE_NAME_BY_ID


STATE_COLORS = {
    "NC": "#6f91bd",
    "PT": "#e49a78",
    "LN": "#7b9e69",
    "ML": "#b07aa1",
    "MO": "#d6a64f",
    "MP": "#72a7a1",
}
FORMATS = ("pdf", "svg", "png")


def load_parameter_estimates(
    result_path: Path,
) -> tuple[
    float,
    tuple[str, ...],
    dict[str, list[float]],
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    estimates: dict[str, list[float]] = {}
    transition_values: dict[tuple[int, int], float] = {}
    tree_sampling_values: dict[int, list[float]] = {}
    tree_transition_values: dict[int, np.ndarray] = {}
    log_likelihood: float | None = None
    in_tree_sampling = False
    in_tree_transitions = False

    for line_number, line in enumerate(result_path.read_text().splitlines(), start=1):
        if line.startswith("LL="):
            log_likelihood = float(line.partition("=")[2])
        elif line == "transition_probabilities_by_tree:":
            in_tree_sampling = False
            in_tree_transitions = True
        elif line.startswith(("transition_times:", "transition_times_by_tree:")):
            in_tree_sampling = False
            in_tree_transitions = False
        elif line == "sampling_by_tree:":
            in_tree_sampling = True
            in_tree_transitions = False
        elif line.startswith(("birth:", "death:", "sampling:")):
            in_tree_sampling = False
            in_tree_transitions = False
            name, _, raw_values = line.partition(":")
            estimates[name] = [float(value) for value in raw_values.split()]
        elif in_tree_sampling and line.startswith("tree_"):
            label, separator, raw_values = line.partition(":")
            if not separator:
                raise ValueError(
                    f"Malformed per-tree sampling estimate at "
                    f"{result_path}:{line_number}"
                )
            tree_index = int(label.removeprefix("tree_"))
            tree_sampling_values[tree_index] = [
                float(value) for value in raw_values.split()
            ]
        elif "->" in line and not in_tree_transitions:
            in_tree_sampling = False
            edge, separator, raw_value = line.partition(" ")
            if not separator:
                raise ValueError(
                    f"Malformed transition estimate at {result_path}:{line_number}"
                )
            source_text, target_text = edge.split("->", maxsplit=1)
            transition_values[(int(source_text), int(target_text))] = float(raw_value)

    if log_likelihood is None or not math.isfinite(log_likelihood):
        raise ValueError(f"{result_path} is missing a finite log likelihood.")
    if set(estimates) != {"birth", "death", "sampling"}:
        raise ValueError(
            f"{result_path} must contain birth, death, and sampling estimates."
        )

    num_states = len(estimates["birth"])
    if num_states == 0 or any(
        len(values) != num_states for values in estimates.values()
    ):
        raise ValueError(f"Parameter vectors in {result_path} have inconsistent sizes.")
    if num_states > len(STATE_NAME_BY_ID) or any(
        not math.isfinite(value)
        for values in estimates.values()
        for value in values
    ):
        raise ValueError(f"Parameter vectors in {result_path} are invalid.")

    state_names = tuple(STATE_NAME_BY_ID[index] for index in range(num_states))
    transition_matrix = np.zeros((num_states, num_states), dtype=float)
    for (source, target), probability in transition_values.items():
        if not (
            0 <= source < num_states
            and 0 <= target < num_states
            and math.isfinite(probability)
            and 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                f"Invalid transition estimate {source}->{target}={probability}."
            )
        transition_matrix[source, target] = probability

    row_sums = transition_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-8, atol=1e-8):
        raise ValueError(
            f"Transition matrix rows in {result_path} must sum to one; "
            f"found {row_sums.tolist()}."
        )
    tree_sampling = None
    if tree_sampling_values:
        expected_tree_indices = list(range(len(tree_sampling_values)))
        if sorted(tree_sampling_values) != expected_tree_indices:
            raise ValueError(
                f"Per-tree sampling indices in {result_path} must be contiguous "
                "from zero."
            )
        if any(
            len(values) not in {1, num_states}
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in values
            )
            for values in tree_sampling_values.values()
        ):
            raise ValueError(
                f"Per-tree sampling values in {result_path} must contain one "
                "or one-per-state valid probabilities per tree."
            )
        tree_sampling = np.asarray(
            [tree_sampling_values[index] for index in expected_tree_indices],
            dtype=float,
        )
        mean_sampling = tree_sampling.mean(axis=0)
        if tree_sampling.shape[1] == 1:
            mean_sampling = np.repeat(mean_sampling, num_states)
        if not np.allclose(mean_sampling, estimates["sampling"], rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"Per-tree sampling mean does not match the sampling vector "
                f"in {result_path}."
            )

    transition_header = "transition_probabilities_by_tree:"
    if transition_header in result_path.read_text().splitlines():
        lines = result_path.read_text().splitlines()
        current_tree: int | None = None
        for line in lines[lines.index(transition_header) + 1 :]:
            if line in {"transition_times:", "transition_times_by_tree:"}:
                break
            tree_match = re.fullmatch(r"tree_(\d+):", line)
            if tree_match:
                current_tree = int(tree_match.group(1))
                if current_tree in tree_transition_values:
                    raise ValueError(
                        f"Duplicate per-tree transition matrix in {result_path}"
                    )
                tree_transition_values[current_tree] = np.zeros(
                    (num_states, num_states),
                    dtype=float,
                )
                continue
            transition_match = re.fullmatch(
                r"(\d+)->(\d+)\s+([0-9.eE+-]+)",
                line,
            )
            if transition_match is None:
                if line.strip():
                    raise ValueError(
                        f"Malformed per-tree transition value in {result_path}: {line}"
                    )
                continue
            if current_tree is None:
                raise ValueError(
                    f"Per-tree transition value precedes tree label in {result_path}"
                )
            source, target, value = transition_match.groups()
            tree_transition_values[current_tree][int(source), int(target)] = float(value)
        if sorted(tree_transition_values) != list(range(len(tree_transition_values))):
            raise ValueError(
                f"Per-tree transition indices in {result_path} must be contiguous."
            )
    tree_transitions = (
        np.asarray(
            [tree_transition_values[index] for index in range(len(tree_transition_values))],
            dtype=float,
        )
        if tree_transition_values
        else None
    )
    if tree_transitions is not None and not np.allclose(
        tree_transitions.mean(axis=0),
        transition_matrix,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"Per-tree transition mean does not match shared transition summary "
            f"in {result_path}"
        )

    return (
        log_likelihood,
        state_names,
        estimates,
        transition_matrix,
        tree_sampling,
        tree_transitions,
    )


def _draw_tree_sampling_heatmap(
    axis: plt.Axes,
    state_names: tuple[str, ...],
    sampling: np.ndarray,
) -> None:
    labels = ("Shared",) if sampling.shape[1] == 1 else state_names
    image = axis.imshow(
        sampling,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    for tree_index, row in enumerate(sampling):
        for state_index, value in enumerate(row):
            axis.text(
                state_index,
                tree_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "#202124",
                fontsize=10,
            )
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(sampling)), labels=[f"Tree {i}" for i in range(len(sampling))])
    axis.set_xlabel(
        "Sampling shared across cell types"
        if sampling.shape[1] == 1
        else "Cell type",
        fontsize=12,
    )
    axis.set_ylabel("Tree", fontsize=12)
    axis.set_title("Per-tree sampling probabilities", fontsize=14, pad=5)
    axis.tick_params(axis="both", labelsize=10, length=0, pad=3)
    axis.spines[:].set_visible(False)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Sampling probability", fontsize=10)
    colorbar.ax.tick_params(labelsize=9, length=2)


def _style_axis(axis: plt.Axes, title: str, y_label: str) -> None:
    axis.set_title(title, fontsize=14, fontweight="normal", pad=5)
    axis.set_ylabel(y_label, fontsize=13, labelpad=2)
    axis.tick_params(axis="both", labelsize=12, length=3, pad=2)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)


def _draw_state_bars(
    axis: plt.Axes,
    state_names: tuple[str, ...],
    values: list[float],
    title: str,
    y_label: str,
) -> None:
    positions = np.arange(len(state_names))
    bars = axis.bar(
        positions,
        values,
        color=[STATE_COLORS[state] for state in state_names],
        edgecolor="#303030",
        linewidth=0.5,
        width=0.64,
        zorder=3,
    )
    axis.axhline(0.0, color="#303030", linewidth=0.8, zorder=2)
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.3g}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4 if value >= 0.0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=11,
        )
    axis.set_xticks(positions, labels=state_names)
    span = max(values) - min(values)
    padding = 0.12 * (span if span > 0.0 else max(abs(max(values)), 1.0))
    axis.set_ylim(min(0.0, min(values) - padding), max(0.0, max(values) + padding))
    _style_axis(axis, title, y_label)


def _draw_transition_heatmap(
    axis: plt.Axes,
    state_names: tuple[str, ...],
    matrix: np.ndarray,
) -> None:
    positive = matrix[matrix > 0.0]
    if positive.size == 0:
        raise ValueError("Transition matrix has no positive probabilities.")
    lower = float(positive.min())
    upper = float(positive.max())
    if lower == upper:
        lower = upper / 10.0
    image = axis.imshow(
        np.ma.masked_where(matrix <= 0.0, matrix),
        cmap="Blues",
        norm=LogNorm(vmin=lower, vmax=upper),
        interpolation="nearest",
    )
    for source, source_name in enumerate(state_names):
        for target, target_name in enumerate(state_names):
            probability = matrix[source, target]
            normalized = image.norm(probability) if probability > 0.0 else 0.0
            axis.text(
                target,
                source,
                f"{probability:.2g}" if probability > 0.0 else "0",
                ha="center",
                va="center",
                color="white" if normalized > 0.62 else "#202124",
                fontsize=11,
            )
    axis.set_xticks(range(len(state_names)), labels=state_names)
    axis.set_yticks(range(len(state_names)), labels=state_names)
    axis.set_xlabel("Destination", fontsize=12, labelpad=3)
    axis.set_ylabel("Source", fontsize=12, labelpad=3)
    axis.set_title("Transition probabilities", fontsize=14, pad=5)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(-0.5, len(state_names), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(state_names), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", labelsize=11, length=0, pad=2)
    axis.spines[:].set_visible(False)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Transition probability (log scale)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9, length=2)


def _draw_tree_transition_heatmaps(
    axes: np.ndarray,
    state_names: tuple[str, ...],
    matrices: np.ndarray,
    tree_names: tuple[str, ...] = (),
) -> None:
    for tree_index, axis in enumerate(axes.flat):
        if tree_index >= len(matrices):
            axis.set_visible(False)
            continue
        _draw_transition_heatmap(
            axis,
            state_names,
            matrices[tree_index],
        )
        label = (
            tree_names[tree_index]
            if tree_index < len(tree_names)
            else f"Tree {tree_index}"
        )
        axis.set_title(f"{label} transition probabilities", fontsize=11)


def _draw_relative_sampling_probabilities(
    state_names: tuple[str, ...],
    tree_sampling: np.ndarray,
    tree_names: tuple[str, ...] = (),
) -> plt.Figure:
    reference_index = state_names.index("NC")
    target_states = [name for name in state_names if name != "NC"]
    columns = 3
    rows = int(math.ceil(len(tree_sampling) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 7.0),
        sharey=True,
        squeeze=False,
    )
    for tree_index, axis in enumerate(axes.flat):
        if tree_index >= len(tree_sampling):
            axis.set_visible(False)
            continue
        for state_index, target_state in enumerate(target_states):
            target_state_index = state_names.index(target_state)
            ratio = (
                tree_sampling[tree_index, target_state_index]
                / tree_sampling[tree_index, reference_index]
            )
            if not math.isfinite(float(ratio)) or ratio <= 0.0:
                raise ValueError(
                    f"Relative sampling probability for tree {tree_index}, "
                    f"{target_state}/NC must be finite and positive."
                )
            axis.scatter(
                [state_index],
                [ratio],
                s=48,
                color=STATE_COLORS[target_state],
                zorder=3,
            )
            axis.annotate(
                f"{ratio:.3g}",
                (state_index, ratio),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        label = (
            tree_names[tree_index]
            if tree_index < len(tree_names)
            else f"Tree {tree_index}"
        )
        axis.set_title(label, fontsize=12, fontweight="normal")
        axis.set_xticks(
            range(len(target_states)),
            labels=[f"{state}/NC" for state in target_states],
        )
        axis.set_xlim(-0.5, len(target_states) - 0.5)
        axis.set_yscale("log")
        axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)
    figure.supylabel(r"Relative sampling probability $\rho_s/\rho_{NC}$")
    figure.suptitle("Relative sampling probabilities by tree", fontsize=14, y=0.99)
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))
    return figure


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in FORMATS:
        output = Path(f"{output_stem}.{extension}")
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01, dpi=200)
        outputs.append(output)
    plt.close(figure)
    return outputs


def plot_combined_inference(
    result_path: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    (
        log_likelihood,
        state_names,
        estimates,
        transition_matrix,
        tree_sampling,
        tree_transitions,
    ) = (
        load_parameter_estimates(result_path)
    )
    sampling = estimates["sampling"]
    scaled_birth = [
        birth * sample
        for birth, sample in zip(estimates["birth"], sampling)
    ]
    output_dir = output_dir or (
        Path("outputs/sgootr/combined_inference") / result_path.parent.name
    )
    base = result_path.stem

    with matplotlib.rc_context(
        {"font.family": "Arial", "mathtext.fontset": "stixsans"}
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(14.0, 4.2),
            gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]},
        )
        _draw_state_bars(
            axes[0],
            state_names,
            [
                birth - death
                for birth, death in zip(estimates["birth"], estimates["death"])
            ],
            "Net growth rate",
            r"$\beta_s - \delta_s$",
        )
        _draw_transition_heatmap(axes[1], state_names, transition_matrix)
        _draw_state_bars(
            axes[2],
            state_names,
            scaled_birth,
            "Scaled birth rates",
            r"$\beta_s \, \rho_s$",
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.995,
            top=0.94,
            bottom=0.17,
            wspace=0.34,
        )
        outputs = _save_figure(figure, output_dir / f"{base}_composite")

        for parameter, title, y_label in (
            ("birth", "Birth rates", r"$\beta_s$"),
            ("death", "Death rates", r"$\delta_s$"),
            ("sampling", "Sampling rates", r"$\rho_s$"),
        ):
            figure, axis = plt.subplots(figsize=(6.5, 4.2))
            _draw_state_bars(
                axis,
                state_names,
                estimates[parameter],
                title,
                y_label,
            )
            figure.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.19)
            outputs.extend(_save_figure(figure, output_dir / f"{base}_{parameter}"))

        figure, axis = plt.subplots(figsize=(6.5, 4.8))
        _draw_transition_heatmap(axis, state_names, transition_matrix)
        figure.tight_layout()
        outputs.extend(_save_figure(figure, output_dir / f"{base}_transitions"))

        if tree_transitions is not None:
            rows = int(math.ceil(len(tree_transitions) / 3))
            figure, axes = plt.subplots(
                rows,
                3,
                figsize=(10.0, 3.2 * rows),
                squeeze=False,
            )
            tree_names_path = result_path.parent / "trees.txt"
            tree_names = (
                tuple(
                    line.strip()
                    for line in tree_names_path.read_text().splitlines()
                    if line.strip()
                )
                if tree_names_path.is_file()
                else ()
            )
            _draw_tree_transition_heatmaps(
                axes,
                state_names,
                tree_transitions,
                tree_names,
            )
            figure.tight_layout()
            outputs.extend(
                _save_figure(
                    figure,
                    output_dir / "plot_2_all_tree_transition_matrices",
                )
            )

        if tree_sampling is not None:
            figure, axis = plt.subplots(figsize=(6.5, 4.8))
            _draw_tree_sampling_heatmap(axis, state_names, tree_sampling)
            figure.tight_layout()
            outputs.extend(_save_figure(figure, output_dir / f"{base}_sampling_by_tree"))

            if tree_sampling.shape[1] == len(state_names):
                tree_names_path = result_path.parent / "trees.txt"
                tree_names = (
                    tuple(
                        line.strip()
                        for line in tree_names_path.read_text().splitlines()
                        if line.strip()
                    )
                    if tree_names_path.is_file()
                    else ()
                )
                figure = _draw_relative_sampling_probabilities(
                    state_names,
                    tree_sampling,
                    tree_names,
                )
                outputs.extend(
                    _save_figure(
                        figure,
                        output_dir
                        / f"{base}_relative_sampling_probability_estimates_by_tree",
                    )
                )

    print(f"Log likelihood: {log_likelihood:.15g}")
    for output in outputs:
        print(f"Wrote {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot state-specific estimates from a combined SGOOTR inference."
    )
    parser.add_argument("result", type=Path, help="Inference result text file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to a figures/ folder beside the result.",
    )
    arguments = parser.parse_args()
    plot_combined_inference(arguments.result, arguments.output_dir)


if __name__ == "__main__":
    main()
