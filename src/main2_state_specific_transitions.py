from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs, read_free_edges
from src.mle_state_specific_transitions import (
    EPOCH_EDGES,
    STATE_NAMES,
    optimize_parameters,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


def _read_events(
    nc_to_pt_time: float,
    pt_to_ln_time: float,
    second_pt_to_ln_time: float | None = None,
) -> tuple[list[float], tuple[tuple[int, int], ...]]:
    events = [
        (nc_to_pt_time, EPOCH_EDGES[0]),
        (pt_to_ln_time, EPOCH_EDGES[1]),
    ]
    if second_pt_to_ln_time is not None:
        events.append((second_pt_to_ln_time, EPOCH_EDGES[1]))
    if any(not math.isfinite(value) or value < 0.0 for value, _ in events):
        raise ValueError("Transition times must be finite and nonnegative.")
    if len({value for value, _ in events}) != len(events):
        raise ValueError("All state-specific transition times must be distinct.")
    ordered_events = sorted(events)
    return (
        [time_value for time_value, _ in ordered_events],
        tuple(edge for _, edge in ordered_events),
    )


def _write_result(path: Path, result, epoch_edges: tuple[tuple[int, int], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        output.write(f"LL={result.log_likelihood.item():.15g}\n")
        for label, values in (
            ("birth", result.birth_rates),
            ("death", result.death_rates),
            ("sampling", result.sampling_probabilities),
        ):
            output.write(
                f"{label}: "
                + " ".join(f"{value:.15g}" for value in values.detach().cpu().tolist())
                + "\n"
            )
        output.write("transition_matrices_by_epoch:\n")
        matrices = result.transition_probabilities.detach().cpu()
        for epoch_index, (source, target) in enumerate(epoch_edges):
            output.write(
                f"epoch_{epoch_index} {STATE_NAMES[source]}->{STATE_NAMES[target]}\n"
            )
            for state_index, row in enumerate(matrices[epoch_index].tolist()):
                for target_index, probability in enumerate(row):
                    if probability > 0.0:
                        output.write(
                            f"{STATE_NAMES[state_index]}->{STATE_NAMES[target_index]} "
                            f"{probability:.15g}\n"
                        )
        output.write("transition_times_by_edge:\n")
        for (source, target), time_value in zip(
            epoch_edges,
            result.transition_times.detach().cpu().tolist(),
        ):
            output.write(
                f"{STATE_NAMES[source]}->{STATE_NAMES[target]} {time_value:.15g}\n"
            )


def run(
    input_dir: Path,
    output: Path,
    log: Path,
    nc_to_pt_time: float,
    pt_to_ln_time: float,
    *,
    second_pt_to_ln_time: float | None = None,
    device: str = "cpu",
    lbfgs_max_iter: int = 20,
):
    transition_times, epoch_edges = _read_events(
        nc_to_pt_time,
        pt_to_ln_time,
        second_pt_to_ln_time,
    )
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {input_dir}")

    edge_file = input_dir / "transition_edges.txt"
    if not edge_file.is_file():
        raise FileNotFoundError(f"Transition graph file does not exist: {edge_file}")
    configured_edges = read_free_edges(str(edge_file))
    if set(configured_edges) != set(EPOCH_EDGES) or len(configured_edges) != len(
        EPOCH_EDGES
    ):
        raise ValueError(
            f"{edge_file} must define exactly the NC->PT and PT->LN edges "
            f"{list(EPOCH_EDGES)}; found {configured_edges}"
        )

    tree_pairs = find_tree_file_pairs(input_dir, num_trees=1)
    trees = []
    datastructures = []
    for node_path, edge_path in tree_pairs:
        tree = prepare_nx_tree(
            load_nodes(str(node_path)),
            load_edges(str(edge_path)),
        )
        root_time = float(tree.nodes[tree.graph["root"]]["T"])
        if transition_times[-1] >= root_time:
            raise ValueError(
                f"Both transition times must precede root time {root_time:.15g}."
            )
        levels, hierarchy = cut_tree(tree, transition_times)
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            transition_times,
            num_states=len(STATE_NAMES),
        )
        trees.append(tree)
        datastructures.append(datastructure)

    result = optimize_parameters(
        trees,
        datastructures,
        transition_times,
        epoch_edges=epoch_edges,
        logfile=log,
        device=device,
        lbfgs_max_iter=lbfgs_max_iter,
    )
    _write_result(output, result, epoch_edges)
    print(f"Log likelihood: {result.log_likelihood.item():.15g}")
    print(
        "Transition times: "
        f"NC->PT={nc_to_pt_time:.15g}, "
        f"PT->LN={pt_to_ln_time:.15g}"
        + (
            ""
            if second_pt_to_ln_time is None
            else f", PT->LN={second_pt_to_ln_time:.15g}"
        )
    )
    print(f"Result: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit CRC11 with one NC->PT epoch followed by one PT->LN epoch. "
            "At each epoch, all other states are constrained to remain unchanged."
        )
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nc-to-pt-time", type=float, required=True)
    parser.add_argument("--pt-to-ln-time", type=float, required=True)
    parser.add_argument("--second-pt-to-ln-time", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    args = parser.parse_args()
    run(
        args.directory,
        args.output,
        args.log,
        args.nc_to_pt_time,
        args.pt_to_ln_time,
        second_pt_to_ln_time=args.second_pt_to_ln_time,
        device=args.device,
        lbfgs_max_iter=args.lbfgs_max_iter,
    )


if __name__ == "__main__":
    main()
