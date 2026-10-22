# main2.py

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path


import os
import torch
import networkx as nx

from src.dataprocessor import load_nodes, load_edges, load_transition_times
from src.tree_utils import cut_tree, prepare_nx_tree, get_datastructure_for_likelihood_calculation
from src.process_data.sgootr.tree import STATE_ID_BY_NAME

from src.mle2_given_transitions import (
    optimize_parameters,
)
from src.mle_given_transitions import optimize_parameters as optimize_per_edge_parameters

# Enable this only while diagnosing NaNs or invalid backward operations.
torch.autograd.set_detect_anomaly(False)

# ============================================================
# HELPERS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch phylodynamic likelihood optimization"
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing paired nodes and edges CSV files.",
    )

    parser.add_argument(
        "--num-states",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-trees",
        type=int,
        default=None,
        help="Number of node/edge pairs to load; defaults to all complete pairs.",
    )

    parser.add_argument(
        "--transition-edges-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--transition-times-file",
        type=str,
        default="transition.txt",
        help="File containing one transition time per line.",
    )

    parser.add_argument(
        "--per-edge-transition-times-pattern",
        type=str,
        default=None,
        help=(
            "CSV path pattern for observed per-edge transitions; use {index} to pair "
            "files with each nodes/edges tree, e.g. transitions_{index}_adjusted.csv."
        ),
    )

    parser.add_argument(
        "--num-transition-epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prior",
        action="store_true",
    )

    parser.add_argument(
        "--normalize-root-likelihood",
        action="store_true",
        help="Normalize each tree likelihood by 1 - U(root time).",
    )

    parser.add_argument(
        "--no-stem",
        action="store_true",
        help=(
            "Use crown conditioning for a stem-free binary root, assuming "
            "the known root state is 0."
        ),
    )

    parser.add_argument(
        "--shared-sampling-rate",
        action="store_true",
        help="Use one learned sampling probability shared by all states.",
    )

    parser.add_argument(
        "--fixed-shared-sampling-probability",
        type=float,
        default=None,
        help="Fix the sampling probability to this value for every state.",
    )

    # --------------------------------------------------------
    # PyTorch-specific options
    # --------------------------------------------------------

    parser.add_argument(
        "--optimizer",
        type=str,
        default="lbfgs",
        choices=["adam", "lbfgs"],
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=500,
        help="Stop Adam after this many steps without objective improvement.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, mps, or cpu. Default: automatically detect.",
    )

    return parser.parse_args()


def find_tree_file_pairs(
    directory: str | Path, num_trees: int | None = None
) -> list[tuple[Path, Path]]:
    if num_trees is not None and num_trees <= 0:
        raise ValueError("num_trees must be positive")

    search_root = Path(directory)
    if not search_root.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {search_root}")

    def natural_path_key(path: Path):
        return tuple(
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.relative_to(search_root).as_posix())
        )

    edge_paths = sorted(
        (
            path
            for path in search_root.rglob("*.csv")
            if re.search(r"edges", path.name, flags=re.IGNORECASE)
        ),
        key=natural_path_key,
    )

    pairs = []
    unmatched_edges = []
    for edge_path in edge_paths:
        node_name, replacements = re.subn(
            r"edges",
            "nodes",
            edge_path.name,
            count=1,
            flags=re.IGNORECASE,
        )
        node_path = edge_path.with_name(node_name)
        if replacements == 1 and node_path.is_file():
            pairs.append((node_path, edge_path))
        else:
            unmatched_edges.append(edge_path)

    if not pairs:
        raise RuntimeError(f"No complete nodes/edges pairs found in {search_root}")

    if num_trees is not None and len(pairs) < num_trees:
        unmatched = ", ".join(str(path) for path in unmatched_edges[:5])
        raise RuntimeError(
            f"Expected {num_trees} complete nodes/edges pairs in {search_root}, "
            f"but found {len(pairs)}. Unmatched edge files: {unmatched or 'none'}"
        )

    return pairs if num_trees is None else pairs[:num_trees]


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def read_free_edges(file_path: str) -> list[tuple[int, int]]:
    """
    Equivalent to the C++:

        while (in >> u >> v) {
            free_edges.emplace_back(u, v);
        }

    Therefore whitespace-separated pairs are accepted.
    """

    path = Path(file_path)

    if not path.exists():
        raise RuntimeError(f"Could not open file: {file_path}")

    free_edges: list[tuple[int, int]] = []

    with path.open("r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                raise RuntimeError(
                    f"Malformed transition edge at " f"{file_path}:{line_number}"
                )

            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError as exc:
                raise RuntimeError(
                    f"Malformed transition edge at "
                    f"{file_path}:{line_number}: {line}"
                ) from exc

            free_edges.append((u, v))

    return free_edges


def read_transition_times(file_path: str) -> list[float]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transition-times file does not exist: {path}")

    transition_times = []
    with path.open() as transition_file:
        for line_number, line in enumerate(transition_file, start=1):
            value = line.split("#", 1)[0].strip()
            if not value:
                continue
            try:
                transition_time = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid transition time at {path}:{line_number}: {value}"
                ) from exc
            if not math.isfinite(transition_time) or transition_time < 0:
                raise ValueError(
                    f"Transition time at {path}:{line_number} must be finite "
                    "and nonnegative."
                )
            transition_times.append(transition_time)

    if not transition_times:
        raise ValueError(f"No transition times found in {path}")

    transition_times.sort()
    if any(
        current <= previous
        for previous, current in zip(transition_times, transition_times[1:])
    ):
        raise ValueError(f"Transition times in {path} must be unique.")

    return transition_times


def load_per_edge_transition_data(
    file_path: str | Path,
) -> dict[tuple[int, int], list[tuple[int, int, float]]]:
    transition_df = load_transition_times(str(file_path))
    columns = set(transition_df.columns)
    inference_columns = {"parent", "child", "s0", "s1", "t"}
    pipeline_columns = {
        "parent",
        "child",
        "state_0",
        "state_1",
        "transition_time",
    }
    if inference_columns.issubset(columns):
        use_pipeline_columns = False
    elif pipeline_columns.issubset(columns):
        use_pipeline_columns = True
    else:
        raise ValueError(
            f"{file_path} must contain either "
            f"{', '.join(sorted(inference_columns))} or "
            f"{', '.join(sorted(pipeline_columns))}."
        )

    transitions_by_edge: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for row in transition_df.itertuples(index=False):
        edge = (int(row.parent), int(row.child))
        if use_pipeline_columns:
            try:
                state_0 = STATE_ID_BY_NAME[row.state_0]
                state_1 = STATE_ID_BY_NAME[row.state_1]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown transition state in {file_path}: {exc.args[0]!r}."
                ) from exc
            transition_time = float(row.transition_time)
        else:
            state_0 = int(row.s0)
            state_1 = int(row.s1)
            transition_time = float(row.t)
        event = (state_0, state_1, transition_time)
        transitions_by_edge.setdefault(edge, []).append(event)

    for events in transitions_by_edge.values():
        events.sort(key=lambda event: event[2], reverse=True)

    return transitions_by_edge


def _validate_no_stem_tree(tree: nx.DiGraph) -> None:
    root = tree.graph["root"]
    daughter_edges = list(tree.successors(root))
    if len(daughter_edges) != 2:
        raise ValueError(
            "--no-stem requires a binary crown root with exactly two daughter "
            f"edges; found {len(daughter_edges)}."
        )


def write_result(
    output_file: str,
    result,
    num_states: int,
):
    """
    Write the optimization result in the same general format
    as the C++ main file.
    """

    path = Path(output_file)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    birth = result.birth_rates.detach().cpu().tolist()

    death = result.death_rates.detach().cpu().tolist()

    sampling = result.sampling_probabilities.detach().cpu().tolist()

    transition_probabilities = result.transition_probabilities.detach().cpu().tolist()

    transition_times = result.transition_times.detach().cpu().tolist()

    log_likelihood = result.log_likelihood

    if isinstance(
        log_likelihood,
        torch.Tensor,
    ):
        log_likelihood = log_likelihood.item()

    with path.open("w") as out:

        out.write(f"LL={log_likelihood:.15f}\n")

        out.write("birth: ")

        for x in birth:
            out.write(f"{x:.15g} ")

        out.write("\n")

        out.write("death: ")

        for x in death:
            out.write(f"{x:.15g} ")

        out.write("\n")

        out.write("sampling: ")

        for x in sampling:
            out.write(f"{x:.15g} ")

        out.write("\n")

        out.write("transitions:\n")

        for i in range(num_states):

            for j in range(num_states):

                p = transition_probabilities[i][j]

                if p > 0.0:

                    out.write(f"{i}->{j} " f"{p:.15g}\n")

        out.write("transition_times:\n")

        for k, t in enumerate(transition_times):

            out.write(f"{k} {t:.15g}\n")

        out.write("\n")


# ============================================================
# MAIN
# ============================================================


def main(normalize_root_likelihood: bool | None = None):

    args = parse_args()
    if args.fixed_shared_sampling_probability is not None:
        if not math.isfinite(args.fixed_shared_sampling_probability) or not (
            0.0 < args.fixed_shared_sampling_probability < 1.0
        ):
            raise ValueError(
                "--fixed-shared-sampling-probability must be strictly between "
                "0 and 1."
            )
        if args.shared_sampling_rate:
            raise ValueError(
                "--fixed-shared-sampling-probability cannot be combined with "
                "--shared-sampling-rate."
            )

    if normalize_root_likelihood is None:
        normalize_root_likelihood = args.normalize_root_likelihood
    if args.no_stem and normalize_root_likelihood:
        raise ValueError(
            "--no-stem crown conditioning cannot be combined with "
            "--normalize-root-likelihood."
        )

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    num_states = args.num_states
    num_trees = args.num_trees

    output_file = args.output
    log_file = args.log

    device = get_device(args.device)

    print(f"directory={args.directory}")

    print(f"num_states={num_states}")

    print(f"num_trees={num_trees if num_trees is not None else 'all complete pairs'}")

    print(f"output={output_file}")

    print(f"logfile={log_file}")

    print(f"prior=" f"{'true' if args.prior else 'false'}")

    print(f"device={device}")
    if args.no_stem:
        print("root_conditioning=crown, known_root_state=0, both_daughters_survive")

    if args.transition_edges_file:
        print("transition_edges_file=" f"{args.transition_edges_file}")

    print("STARTED ...")

    start = time.perf_counter()

    try:

        # ====================================================
        # LOAD TREES
        # ====================================================

        trees = []

        tree_file_pairs = find_tree_file_pairs(args.directory, num_trees)

        per_edge_transition_data: dict[
            int, dict[tuple[int, int], list[tuple[int, int, float]]]
        ] = {}
        tree_indices = []
        for tree_position, (node_path, _) in enumerate(tree_file_pairs):
            match = re.search(r"nodes_(\d+)", node_path.stem, flags=re.IGNORECASE)
            tree_index = int(match.group(1)) if match else tree_position
            tree_indices.append(tree_index)

        if args.per_edge_transition_times_pattern:
            if (
                len(tree_file_pairs) > 1
                and "{index}" not in args.per_edge_transition_times_pattern
            ):
                raise ValueError(
                    "The per-edge transition CSV pattern must contain {index} "
                    "when loading multiple trees."
                )

            for tree_index in tree_indices:
                transition_path = args.per_edge_transition_times_pattern.format(
                    index=tree_index
                )
                per_edge_transition_data[tree_index] = load_per_edge_transition_data(
                    transition_path
                )
                event_count = sum(
                    len(events)
                    for events in per_edge_transition_data[tree_index].values()
                )
                print(
                    f"Loaded {event_count} per-edge transitions across "
                    f"{len(per_edge_transition_data[tree_index])} edges "
                    f"from {transition_path}"
                )

        for i, (node_path, edge_path) in enumerate(tree_file_pairs):

            tree_start = time.perf_counter()

            loaded_nodes = load_nodes(str(node_path))

            loaded_edges = load_edges(str(edge_path))

            if len(loaded_nodes) == 0:
                raise RuntimeError(f"No nodes loaded from " f"{node_path}")

            if len(loaded_edges) == 0:
                raise RuntimeError(f"No edges loaded from " f"{edge_path}")

            tree = prepare_nx_tree(
                loaded_nodes,
                loaded_edges,
            )
            if args.no_stem:
                _validate_no_stem_tree(tree)

            trees.append(tree)

            print(
                f"[timing] tree {i} preparation: "
                f"{time.perf_counter() - tree_start:.3f}s "
                f"({len(loaded_nodes)} nodes, {len(loaded_edges)} edges)",
                flush=True,
            )

        print(f"LOADED {len(trees)} TREES")

        # ====================================================
        # FREE TRANSITION EDGES
        # ====================================================

        if args.transition_edges_file:
            free_edges = read_free_edges(args.transition_edges_file)
        elif per_edge_transition_data:
            free_edges = sorted(
                {
                    (source_state, target_state)
                    for transitions_by_edge in per_edge_transition_data.values()
                    for events in transitions_by_edge.values()
                    for source_state, target_state, _ in events
                }
            )
        else:
            raise ValueError(
                "Provide --transition-edges-file or per-edge transition data "
                "from which to infer allowed state transitions."
            )

        print(f"Loaded {len(free_edges)} " f"free transition edges")

        # ====================================================
        # TRANSITION EPOCHS
        # ====================================================

        transition_times = read_transition_times(args.transition_times_file)
        if transition_times[0] == 0.0:
            transition_times = transition_times[1:]

        if per_edge_transition_data:
            event_times = {
                event[2]
                for transitions_by_edge in per_edge_transition_data.values()
                for events in transitions_by_edge.values()
                for event in events
            }
            unmatched_event_times = [
                event_time
                for event_time in event_times
                if not any(
                    math.isclose(
                        event_time,
                        epoch_time,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    for epoch_time in transition_times
                )
            ]
            if unmatched_event_times:
                raise ValueError(
                    "Per-edge transition times must match an epoch in "
                    f"{args.transition_times_file}; unmatched times: "
                    f"{sorted(unmatched_event_times)}"
                )

        print(f"transition_times={transition_times}")

        tree_transition_times = []
        cut_trees = []
        for tree in trees:
            root_time = float(tree.nodes[tree.graph["root"]]["T"])
            local_transition_times = [
                transition_time
                for transition_time in transition_times
                if transition_time < root_time
            ]
            tree_transition_times.append(local_transition_times)
            cut_trees.append(cut_tree(tree, local_transition_times))

        for ct in cut_trees:
            levels, hierarchy = ct
            for i, lv in enumerate(levels):
                print(f"LEVEL {i}")
                for T in lv:
                    print(f"  Tree {T.graph['root']} {T.graph['cut_time']} {len(T.nodes())}")
                
            for u, v in hierarchy.edges():
                print(f"  {u} -> {v}")


        dats = []
        for tree_index, ((levels, hierarchy), local_transition_times) in zip(
            tree_indices, zip(cut_trees, tree_transition_times)
        ):
            dats.append(
                get_datastructure_for_likelihood_calculation(
                    levels,
                    hierarchy,
                    local_transition_times,
                    num_states,
                    per_edge_transition_data=per_edge_transition_data.get(tree_index),
                )
            )

        # ====================================================
        # PRIOR
        # ====================================================

        if args.prior:

            factor = 200.0

            print(f"Using linear prior for " f"transition times with factor {factor}")

        else:

            print(f"No prior for transition times")

            factor = 0.0


        initial_birth_rates = None
        initial_death_rates = None
        initial_sampling_probabilities = None
        initial_transition_probabilities = None

        # ====================================================
        # FIXED PARAMETERS
        # ====================================================

        given_birth_rates = None
        # given_death_rates = [0.2, 0.5]

        # Same as:
        #
        # given_sampling_probabilities = {0.5, 0.6};

        # ====================================================
        # OPTIMIZATION
        # ====================================================

        result = optimize_parameters(
            trees=trees,
            datastructure=dats,
            num_states=num_states,
            given_transition_times=(transition_times),
            free_edges=free_edges,
            logfile=log_file,
            initial_birth_rates=(initial_birth_rates),
            initial_death_rates=(initial_death_rates),
            initial_sampling_probabilities=(initial_sampling_probabilities),
            given_sampling_probabilities=(
                [args.fixed_shared_sampling_probability] * num_states
                if args.fixed_shared_sampling_probability is not None
                else None
            ),
            initial_transition_probabilities=(initial_transition_probabilities),
            # given_death_rates=(given_death_rates),
            # given_birth_rates=(given_birth_rates),
            # given_sampling_probabilities=(given_sampling_probabilities),
            optimizer_name=(args.optimizer),
            learning_rate=(args.learning_rate),
            max_steps=(args.max_steps),
            patience=(args.patience),
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            no_stem=args.no_stem,
            shared_sampling_rate=args.shared_sampling_rate,
        )
        # ====================================================
        # PRINT RESULT
        # ====================================================

        final_ll = result.log_likelihood

        if isinstance(
            final_ll,
            torch.Tensor,
        ):
            final_ll = final_ll.item()

        print(f"Best LL = " f"{final_ll:.15f}")

        print(
            "birth:",
            result.birth_rates.detach().cpu().numpy(),
        )

        print(
            "death:",
            result.death_rates.detach().cpu().numpy(),
        )

        print(
            "sampling:",
            result.sampling_probabilities.detach().cpu().numpy(),
        )

        print("transitions:")

        Q = result.transition_probabilities.detach().cpu().numpy()

        for i in range(num_states):

            for j in range(num_states):

                if Q[i, j] > 0.0:

                    print(f"{i}->{j} " f"{Q[i, j]:.15f}")

        print(
            "transition_times:",
            result.transition_times.detach().cpu().numpy(),
        )

        # ====================================================
        # WRITE RESULT
        # ====================================================

        write_result(
            output_file=output_file,
            result=result,
            num_states=num_states,
        )

        print(f"Results written to " f"{output_file}")

        # ====================================================
        # TIMING
        # ====================================================

        elapsed = time.perf_counter() - start

        print(f"\nTotal execution time: " f"{elapsed:.4f} sec")

    except Exception as exc:

        print(f"Fatal error: {exc}")

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
