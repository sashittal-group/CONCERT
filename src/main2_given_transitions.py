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
        "--per-tree-transition-edges-file",
        type=str,
        default=None,
        help=(
            "Tree-name keyed transition graph file; required entries are "
            "labels such as CRC04: followed by whitespace-separated state pairs."
        ),
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
        "--verbose",
        action="store_true",
        help="Print detailed cut-tree likelihood data structures.",
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
        "--per-tree-sampling-rates",
        action="store_true",
        help=(
            "Infer one sampling probability per tree, shared across all states."
        ),
    )

    parser.add_argument(
        "--per-tree-state-sampling-rates",
        action="store_true",
        help=(
            "Infer a separate sampling probability for every tree and state."
        ),
    )

    parser.add_argument(
        "--per-tree-transition-rates",
        action="store_true",
        help=(
            "Use each tree's observed transition epochs and infer a separate "
            "transition matrix for that tree."
        ),
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


def read_per_tree_transition_edges(
    file_path: str | Path,
) -> dict[str, list[tuple[int, int]]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Per-tree transition graph file does not exist: {path}")

    edges_by_tree: dict[str, list[tuple[int, int]]] = {}
    current_tree: str | None = None
    with path.open() as graph_file:
        for line_number, raw_line in enumerate(graph_file, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.endswith(":") and len(line.split()) == 1:
                current_tree = line[:-1]
                if not current_tree or current_tree in edges_by_tree:
                    raise ValueError(
                        f"Invalid or duplicate tree label at {path}:{line_number}"
                    )
                edges_by_tree[current_tree] = []
                continue
            if current_tree is None:
                raise ValueError(
                    f"Transition edge before tree label at {path}:{line_number}"
                )
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed transition edge at {path}:{line_number}: {line}"
                )
            try:
                edge = (int(parts[0]), int(parts[1]))
            except ValueError as error:
                raise ValueError(
                    f"Malformed transition edge at {path}:{line_number}: {line}"
                ) from error
            if edge[0] == edge[1] or edge in edges_by_tree[current_tree]:
                raise ValueError(
                    f"Invalid or duplicate transition edge at {path}:{line_number}"
                )
            edges_by_tree[current_tree].append(edge)

    if not edges_by_tree or any(not edges for edges in edges_by_tree.values()):
        raise ValueError(f"Every tree in {path} must have at least one transition edge")
    return edges_by_tree


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
    named_pipeline_columns = {
        "parent",
        "child",
        "state_0",
        "state_1",
        "transition_time",
    }
    kptracer_columns = {"parent", "child", "state_0", "state_1", "t"}
    if inference_columns.issubset(columns):
        use_named_state_columns = False
        use_kptracer_columns = False
    elif named_pipeline_columns.issubset(columns):
        use_named_state_columns = True
        use_kptracer_columns = False
    elif kptracer_columns.issubset(columns):
        use_named_state_columns = False
        use_kptracer_columns = True
    else:
        raise ValueError(
            f"{file_path} must contain either "
            f"{', '.join(sorted(inference_columns))}, "
            f"{', '.join(sorted(named_pipeline_columns))}, or "
            f"{', '.join(sorted(kptracer_columns))}."
        )

    transitions_by_edge: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for row in transition_df.itertuples(index=False):
        edge = (int(row.parent), int(row.child))
        if use_named_state_columns:
            try:
                state_0 = STATE_ID_BY_NAME[row.state_0]
                state_1 = STATE_ID_BY_NAME[row.state_1]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown transition state in {file_path}: {exc.args[0]!r}."
                ) from exc
            transition_time = float(row.transition_time)
        elif use_kptracer_columns:
            state_0 = int(row.state_0)
            state_1 = int(row.state_1)
            transition_time = float(row.t)
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

    tree_sampling = (
        result.tree_sampling_probabilities.detach().cpu().tolist()
        if result.tree_sampling_probabilities is not None
        else None
    )
    sampling_probabilities = result.sampling_probabilities.detach().cpu()
    sampling = sampling_probabilities.tolist()

    transition_probabilities = result.transition_probabilities.detach().cpu().tolist()
    tree_transition_probabilities = (
        getattr(result, "tree_transition_probabilities", None).detach().cpu().tolist()
        if getattr(result, "tree_transition_probabilities", None) is not None
        else None
    )

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
        if tree_sampling is not None:
            per_state = isinstance(tree_sampling[0], list)
            out.write(
                "sampling_scope: "
                + (
                    "per_tree_by_state\n"
                    if per_state
                    else "per_tree_shared_across_states\n"
                )
            )
            out.write("sampling_by_tree:\n")
            for tree_index, probabilities in enumerate(tree_sampling):
                values = probabilities if per_state else [probabilities]
                out.write(
                    f"tree_{tree_index}: "
                    + " ".join(f"{value:.15g}" for value in values)
                    + "\n"
                )

        out.write("transitions:\n")

        for i in range(num_states):

            for j in range(num_states):

                p = transition_probabilities[i][j]

                if p > 0.0:

                    out.write(f"{i}->{j} " f"{p:.15g}\n")

        if tree_transition_probabilities is not None:
            out.write("transition_scope: per_tree\n")
            out.write("transition_probabilities_by_tree:\n")
            for tree_index, tree_matrix in enumerate(tree_transition_probabilities):
                out.write(f"tree_{tree_index}:\n")
                for i in range(num_states):
                    for j in range(num_states):
                        probability = tree_matrix[i][j]
                        if probability > 0.0:
                            out.write(f"{i}->{j} {probability:.15g}\n")

        out.write("transition_times:\n")

        for k, t in enumerate(transition_times):

            out.write(f"{k} {t:.15g}\n")

        out.write("\n")
        if getattr(result, "tree_transition_times", None) is not None:
            out.write("transition_times_by_tree:\n")
            for tree_index, tree_times in enumerate(result.tree_transition_times):
                out.write(
                    f"tree_{tree_index}: "
                    + " ".join(f"{value:.15g}" for value in tree_times)
                    + "\n"
                )


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
        if args.per_tree_sampling_rates or args.per_tree_state_sampling_rates:
            raise ValueError(
                "--fixed-shared-sampling-probability cannot be combined with "
                "per-tree sampling-rate options."
            )
    if (
        (args.per_tree_sampling_rates or args.per_tree_state_sampling_rates)
        and args.shared_sampling_rate
    ):
        raise ValueError(
            "Per-tree sampling-rate options cannot be combined with "
            "--shared-sampling-rate."
        )
    if args.per_tree_sampling_rates and args.per_tree_state_sampling_rates:
        raise ValueError(
            "--per-tree-sampling-rates and --per-tree-state-sampling-rates "
            "are mutually exclusive."
        )
    if args.per_tree_transition_rates and not args.per_edge_transition_times_pattern:
        raise ValueError(
            "--per-tree-transition-rates requires "
            "--per-edge-transition-times-pattern."
        )
    if args.per_tree_transition_edges_file and not args.per_tree_transition_rates:
        raise ValueError(
            "--per-tree-transition-edges-file requires "
            "--per-tree-transition-rates."
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
    if args.per_tree_sampling_rates:
        print("sampling_mode=per_tree_shared_across_states")
    elif args.per_tree_state_sampling_rates:
        print("sampling_mode=per_tree_by_state")
    if args.per_tree_transition_rates:
        print("transition_mode=per_tree_observed_times")

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

        tree_free_edges: list[list[tuple[int, int]]] | None = None
        if args.per_tree_transition_edges_file:
            tree_names_path = Path(args.directory) / "trees.txt"
            if not tree_names_path.is_file():
                raise FileNotFoundError(
                    "Per-tree transition graphs require a trees.txt file in "
                    f"{args.directory}"
                )
            tree_names = [
                line.strip()
                for line in tree_names_path.read_text().splitlines()
                if line.strip()
            ]
            if len(tree_names) < len(tree_file_pairs) or len(
                set(tree_names[: len(tree_file_pairs)])
            ) != len(tree_file_pairs):
                raise ValueError(
                    f"{tree_names_path} must contain one unique name per loaded tree"
                )
            tree_names_by_index = {
                tree_index: tree_names[position]
                for position, tree_index in enumerate(tree_indices)
            }
            configured_tree_edges = read_per_tree_transition_edges(
                args.per_tree_transition_edges_file
            )
            missing_graphs = sorted(
                set(tree_names_by_index.values()) - set(configured_tree_edges)
            )
            if missing_graphs:
                raise ValueError(
                    "Per-tree transition graph file is missing trees: "
                    + ", ".join(missing_graphs)
                )
            tree_free_edges = [
                configured_tree_edges[tree_names_by_index[index]]
                for index in tree_indices
            ]
            invalid_edges = [
                edge
                for edges in tree_free_edges
                for edge in edges
                if any(state < 0 or state >= num_states for state in edge)
            ]
            if invalid_edges:
                raise ValueError(
                    f"Per-tree transition graph contains state indices outside "
                    f"0..{num_states - 1}: {invalid_edges}"
                )

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

        if tree_free_edges is not None:
            free_edges = sorted(
                {edge for tree_edges in tree_free_edges for edge in tree_edges}
            )
            if args.transition_edges_file:
                shared_edges = set(read_free_edges(args.transition_edges_file))
                if shared_edges != set(free_edges):
                    raise ValueError(
                        "--transition-edges-file must equal the union of the "
                        "per-tree transition graphs"
                    )
            for tree_index, tree_edges in zip(tree_indices, tree_free_edges):
                observed_edges = {
                    (source, target)
                    for events in per_edge_transition_data.get(tree_index, {}).values()
                    for source, target, _ in events
                }
                disallowed_edges = observed_edges - set(tree_edges)
                if disallowed_edges:
                    raise ValueError(
                        f"Tree {tree_index} has observed transitions not allowed "
                        f"by its graph: {sorted(disallowed_edges)}"
                    )
        elif args.transition_edges_file:
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
        if tree_free_edges is not None:
            for tree_index, tree_edges in zip(tree_indices, tree_free_edges):
                print(f"tree_{tree_index}_transition_edges={tree_edges}")

        # ====================================================
        # TRANSITION EPOCHS
        # ====================================================

        transition_times = read_transition_times(args.transition_times_file)
        if transition_times[0] == 0.0:
            transition_times = transition_times[1:]

        tree_transition_times: list[list[float]] | None = None
        if args.per_tree_transition_rates:
            tree_transition_times = []
            for tree_index in tree_indices:
                if tree_index not in per_edge_transition_data:
                    raise ValueError(
                        "Per-tree transition rates require transition event data "
                        f"for tree {tree_index}."
                    )
                observed_times = sorted(
                    {
                        transition_time
                        for events in per_edge_transition_data[tree_index].values()
                        for _, _, transition_time in events
                        if transition_time > 0.0
                    }
                )
                if not observed_times:
                    raise ValueError(
                        f"Tree {tree_index} has no positive transition event times."
                    )
                tree_transition_times.append(observed_times)
            transition_times = sorted(
                {time for tree_times in tree_transition_times for time in tree_times}
            )
        elif per_edge_transition_data:
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

        cut_trees = []
        for tree in trees:
            root_time = float(tree.nodes[tree.graph["root"]]["T"])
            base_transition_times = (
                tree_transition_times[len(cut_trees)]
                if tree_transition_times is not None
                else transition_times
            )
            local_transition_times = [
                transition_time
                for transition_time in base_transition_times
                if transition_time < root_time
            ]
            if tree_transition_times is not None:
                tree_transition_times[len(cut_trees)] = local_transition_times
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
        datastructure_transition_times = (
            tree_transition_times
            if tree_transition_times is not None
            else [transition_times] * len(cut_trees)
        )
        for tree_index, ((levels, hierarchy), local_transition_times) in zip(
            tree_indices, zip(cut_trees, datastructure_transition_times)
        ):
            dats.append(
                get_datastructure_for_likelihood_calculation(
                    levels,
                    hierarchy,
                    local_transition_times,
                    num_states,
                    per_edge_transition_data=per_edge_transition_data.get(tree_index),
                    verbose=args.verbose,
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
            per_tree_sampling_rates=args.per_tree_sampling_rates,
            per_tree_state_sampling_rates=args.per_tree_state_sampling_rates,
            per_tree_transition_rates=args.per_tree_transition_rates,
            tree_transition_times=tree_transition_times,
            tree_free_edges=tree_free_edges,
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
        if result.tree_sampling_probabilities is not None:
            print("sampling_by_tree:")
            print(
                result.tree_sampling_probabilities.detach().cpu().numpy(),
            )

        print("transitions:")

        Q = result.transition_probabilities.detach().cpu().numpy()

        for i in range(num_states):

            for j in range(num_states):

                if Q[i, j] > 0.0:

                    print(f"{i}->{j} " f"{Q[i, j]:.15f}")

        if result.tree_transition_probabilities is not None:
            print("transitions_by_tree:")
            for tree_index, tree_matrix in enumerate(
                result.tree_transition_probabilities.detach().cpu().numpy()
            ):
                print(f"tree_{tree_index}:")
                for i in range(num_states):
                    for j in range(num_states):
                        if tree_matrix[i, j] > 0.0:
                            print(f"{i}->{j} {tree_matrix[i, j]:.15f}")

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
