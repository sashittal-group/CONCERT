from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs
from src.mle_state_specific_transitions import EPOCH_EDGES, STATE_NAMES
from src.closed_form_state_specific_transitions import (
    calculate_Us,
    calculate_log_likelihood,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


GRID_SIZE = 1000
EPOCH_PATTERN = re.compile(r"epoch_(\d+) ([A-Z]+)->([A-Z]+)")


def _read_fit(path: Path) -> tuple[
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[int, int], ...],
    tuple[float, ...],
]:
    lines = path.read_text().splitlines()
    likelihood_lines = [line for line in lines if line.startswith("LL=")]
    if len(likelihood_lines) != 1:
        raise ValueError(f"Expected one LL entry in {path}.")
    expected_likelihood = float(likelihood_lines[0].partition("=")[2])
    if not math.isfinite(expected_likelihood):
        raise ValueError(f"Likelihood in {path} is not finite.")

    values: dict[str, torch.Tensor] = {}
    for label in ("birth", "death", "sampling"):
        matching = [line for line in lines if line.startswith(f"{label}:")]
        if len(matching) != 1:
            raise ValueError(f"Expected one {label} entry in {path}.")
        values[label] = torch.tensor(
            [float(value) for value in matching[0].partition(":")[2].split()],
            dtype=torch.float64,
        )
        if values[label].shape != (len(STATE_NAMES),):
            raise ValueError(f"Expected three {label} values in {path}.")

    try:
        matrix_start = lines.index("transition_matrices_by_epoch:") + 1
        times_start = lines.index("transition_times_by_edge:")
    except ValueError as error:
        raise ValueError(f"Missing epoch transition data in {path}.") from error

    epoch_edges: list[tuple[int, int]] = []
    matrices: list[torch.Tensor] = []
    for line in lines[matrix_start:times_start]:
        epoch_match = EPOCH_PATTERN.fullmatch(line)
        if epoch_match:
            if len(matrices) > len(epoch_edges):
                raise ValueError(f"Malformed transition matrix section in {path}.")
            source_name, target_name = epoch_match.groups()[1:]
            try:
                edge = (STATE_NAMES.index(source_name), STATE_NAMES.index(target_name))
            except ValueError as error:
                raise ValueError(f"Unknown epoch edge in {path}: {line}") from error
            epoch_edges.append(edge)
            matrices.append(torch.eye(len(STATE_NAMES), dtype=torch.float64))
            continue
        if "->" not in line or not matrices:
            continue
        transition, probability = line.split()
        source_name, target_name = transition.split("->")
        try:
            source = STATE_NAMES.index(source_name)
            target = STATE_NAMES.index(target_name)
        except ValueError as error:
            raise ValueError(f"Unknown transition state in {path}: {line}") from error
        matrices[-1][source, target] = float(probability)

    edge_lines = lines[times_start + 1:]
    event_times: list[tuple[tuple[int, int], float]] = []
    for line in edge_lines:
        if "->" not in line:
            continue
        edge_text, time_text = line.split()
        source_name, target_name = edge_text.split("->")
        try:
            edge = (STATE_NAMES.index(source_name), STATE_NAMES.index(target_name))
        except ValueError as error:
            raise ValueError(f"Unknown event edge in {path}: {line}") from error
        event_times.append((edge, float(time_text)))
    if len(epoch_edges) != len(event_times) or tuple(epoch_edges) != tuple(
        edge for edge, _ in event_times
    ):
        raise ValueError(f"Epoch and transition-time sections do not match in {path}.")
    times = tuple(time for _, time in event_times)
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"Transition times in {path} must be strictly increasing.")
    if not set(epoch_edges).issubset(EPOCH_EDGES):
        raise ValueError(f"Unsupported transition edge in {path}.")
    return (
        expected_likelihood,
        values["birth"],
        values["death"],
        values["sampling"],
        torch.stack(matrices),
        tuple(epoch_edges),
        times,
    )


def _tree(input_dir: Path) -> nx.DiGraph:
    with redirect_stdout(StringIO()):
        pairs = find_tree_file_pairs(input_dir, num_trees=1)
        if len(pairs) != 1:
            raise ValueError(
                f"Expected exactly one tree in {input_dir}; found {len(pairs)}."
            )
        node_path, edge_path = pairs[0]
        return prepare_nx_tree(load_nodes(str(node_path)), load_edges(str(edge_path)))


def _likelihood_at_times(
    tree: nx.DiGraph,
    times: tuple[float, ...],
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
    transition_matrices: torch.Tensor,
) -> float:
    with redirect_stdout(StringIO()), torch.no_grad():
        levels, hierarchy = cut_tree(tree, times)
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            times,
            num_states=len(STATE_NAMES),
        )
        time_tensor = torch.tensor(times, dtype=torch.float64)
        u_result = calculate_Us(
            birth,
            death,
            sampling,
            transition_matrices,
            time_tensor,
            root_times=[
                float(tree.nodes[tree.graph["root"]]["T"])
            ],
        )
        likelihood = calculate_log_likelihood(
            datastructure,
            birth,
            death,
            transition_matrices,
            sampling,
            u_result.Cs,
            times,
        )[0, 0]
    value = float(likelihood.item())
    if math.isnan(value) or value == math.inf:
        raise FloatingPointError(f"Non-finite likelihood at transition times {times}.")
    return value


def _profile_values(
    tree: nx.DiGraph,
    baseline_times: tuple[float, ...],
    epoch_edges: tuple[tuple[int, int], ...],
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
    transition_matrices: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    root_time = float(tree.nodes[tree.graph["root"]]["T"])
    rows: list[dict[str, float | int | str]] = []
    for epoch_index, edge in enumerate(epoch_edges):
        label = f"{STATE_NAMES[edge[0]]}->{STATE_NAMES[edge[1]]}"
        baseline_time = baseline_times[epoch_index]
        for grid_index in range(1, math.ceil(root_time * GRID_SIZE)):
            candidate = grid_index / GRID_SIZE
            if candidate >= root_time:
                continue
            if epoch_index > 0 and candidate <= baseline_times[epoch_index - 1]:
                continue
            if (
                epoch_index + 1 < len(baseline_times)
                and candidate >= baseline_times[epoch_index + 1]
            ):
                continue
            candidate_times = list(baseline_times)
            candidate_times[epoch_index] = candidate
            candidate_tuple = tuple(candidate_times)
            likelihood = _likelihood_at_times(
                tree,
                candidate_tuple,
                birth,
                death,
                sampling,
                transition_matrices,
            )
            rows.append(
                {
                    "epoch_index": epoch_index,
                    "transition_edge": label,
                    "varied_time": candidate,
                    "baseline_time": baseline_time,
                    "log_likelihood": likelihood,
                }
            )
    return rows


def run_profile(
    input_dir: Path,
    inference_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    input_dir = input_dir.resolve()
    inference_path = inference_path.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {input_dir}")
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference result does not exist: {inference_path}")
    expected_outputs = {
        "fixed_parameter_transition_time_profile.csv",
        "fixed_parameter_transition_time_profile.png",
        "fixed_parameter_transition_time_profile.pdf",
    }
    if output_dir.exists() and any(
        path.name not in expected_outputs for path in output_dir.iterdir()
    ):
        raise FileExistsError(f"Profile output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        expected_likelihood,
        birth,
        death,
        sampling,
        transition_matrices,
        epoch_edges,
        baseline_times,
    ) = _read_fit(inference_path)
    tree = _tree(input_dir)
    baseline_likelihood = _likelihood_at_times(
        tree,
        baseline_times,
        birth,
        death,
        sampling,
        transition_matrices,
    )
    if not math.isclose(
        baseline_likelihood,
        expected_likelihood,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "Fixed-parameter baseline likelihood does not reproduce the saved fit: "
            f"{baseline_likelihood:.15g} vs {expected_likelihood:.15g}."
        )

    rows = _profile_values(
        tree,
        baseline_times,
        epoch_edges,
        birth,
        death,
        sampling,
        transition_matrices,
    )
    finite_likelihoods = [
        float(row["log_likelihood"])
        for row in rows
        if math.isfinite(float(row["log_likelihood"]))
    ]
    if not finite_likelihoods:
        raise FloatingPointError("No finite likelihoods were found in the profiles.")
    likelihood_span = max(finite_likelihoods) - min(finite_likelihoods)
    plot_floor = min(finite_likelihoods) - max(1.0, 0.08 * likelihood_span)
    plot_ceiling = max(finite_likelihoods) + max(1.0, 0.03 * likelihood_span)

    csv_path = output_dir / "fixed_parameter_transition_time_profile.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "epoch_index",
                "transition_edge",
                "varied_time",
                "baseline_time",
                "log_likelihood",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        len(epoch_edges),
        1,
        figsize=(8, 3.4 * len(epoch_edges)),
        squeeze=False,
        sharey=True,
    )
    for epoch_index, (axis_row, edge) in enumerate(zip(axes, epoch_edges)):
        axis = axis_row[0]
        epoch_rows = [row for row in rows if row["epoch_index"] == epoch_index]
        axis.plot(
            [row["varied_time"] for row in epoch_rows],
            [row["log_likelihood"] for row in epoch_rows],
            linewidth=1.8,
        )
        zero_likelihood_times = [
            float(row["varied_time"])
            for row in epoch_rows
            if float(row["log_likelihood"]) == -math.inf
        ]
        if zero_likelihood_times:
            axis.scatter(
                zero_likelihood_times,
                [plot_floor] * len(zero_likelihood_times),
                marker="v",
                color="tab:red",
                s=18,
                label="−∞ (zero likelihood)",
            )
        axis.axvline(
            baseline_times[epoch_index],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"fitted time = {baseline_times[epoch_index]:.3f}",
        )
        axis.scatter(
            [baseline_times[epoch_index]],
            [baseline_likelihood],
            color="black",
            s=24,
            zorder=3,
        )
        axis.set_title(
            f"Vary epoch {epoch_index}: "
            f"{STATE_NAMES[edge[0]]}→{STATE_NAMES[edge[1]]}"
        )
        axis.set_xlabel("Varied transition time")
        axis.set_ylabel("Log-likelihood")
        epoch_times = [float(row["varied_time"]) for row in epoch_rows]
        axis.set_xlim(min(epoch_times), max(epoch_times))
        axis.set_ylim(plot_floor, plot_ceiling)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend()
    cohort_label = " ".join(input_dir.name.split("_")[:2])
    figure.suptitle(
        f"{cohort_label}: fixed-parameter transition-time profiles\n"
        "Other event times, rates, sampling, and transition probabilities held fixed",
        y=1.01,
    )
    figure.tight_layout()
    png_path = output_dir / "fixed_parameter_transition_time_profile.png"
    pdf_path = output_dir / "fixed_parameter_transition_time_profile.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return csv_path, png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile each transition time independently while keeping all "
            "other times and fitted model parameters fixed."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in run_profile(args.input_dir, args.inference, args.output_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
