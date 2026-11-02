# main2.py

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path


import os
import torch

from src.dataprocessor import load_nodes, load_edges
from src.tree_utils import cut_tree, prepare_nx_tree, get_datastructure_for_likelihood_calculation

from src.mle_closed_form2 import (
    optimize_parameters,
)

# Enable this only while diagnosing NaNs or invalid backward operations.
torch.autograd.set_detect_anomaly(False)

# ============================================================
# HELPERS
# ============================================================


def parse_args(require_initial_parameters: bool = False):
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
        required=True,
    )

    parser.add_argument(
        "--transition-times-file",
        type=str,
        default="transition.txt",
        help="File containing one transition time per line.",
    )

    parser.add_argument(
        "--num-transition-epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--preserve-zero-transition-time",
        action="store_true",
        help="Treat a leading zero in the transition-time file as a transition.",
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
        "--initial-parameters",
        type=Path,
        required=require_initial_parameters,
        help="Initialize the optimizer from a previous .out or .tmp.out file.",
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

            out.write(f"{k} {t:.17g}\n")

        out.write("\n")


def read_initial_parameters(
    file_path: str | Path,
    num_states: int,
    free_edges: list[tuple[int, int]],
) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Initial parameter file does not exist: {path}")
    lines = path.read_text().splitlines()

    vectors: dict[str, list[float]] = {}
    for name in ("birth", "death", "sampling"):
        matching_lines = [line for line in lines if line.startswith(f"{name}:")]
        if len(matching_lines) != 1:
            raise ValueError(
                f"Expected one {name!r} vector in {path}; "
                f"found {len(matching_lines)}"
            )
        try:
            vector = [
                float(value)
                for value in matching_lines[0].split(":", 1)[1].split()
            ]
        except ValueError as error:
            raise ValueError(f"Invalid {name!r} vector in {path}") from error
        if len(vector) != num_states or not all(math.isfinite(value) for value in vector):
            raise ValueError(
                f"Expected {num_states} finite values in the {name!r} vector "
                f"in {path}"
            )
        vectors[name] = vector
    if any(
        birth < death
        for birth, death in zip(vectors["birth"], vectors["death"])
    ):
        raise ValueError(f"Initial birth rates must be at least death rates in {path}")
    if any(not 0.0 < value < 1.0 for value in vectors["sampling"]):
        raise ValueError(f"Initial sampling rates must be in (0, 1) in {path}")

    transition_values: dict[tuple[int, int], float] = {}
    in_transitions = False
    for line in lines:
        stripped = line.strip()
        if stripped == "transitions:":
            in_transitions = True
            continue
        if stripped == "transition_times:":
            in_transitions = False
            continue
        if not in_transitions or not stripped:
            continue
        match = re.fullmatch(
            r"(\d+)->(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            stripped,
        )
        if match is None:
            continue
        source, target, probability = match.groups()
        edge = (int(source), int(target))
        value = float(probability)
        if edge in transition_values:
            raise ValueError(f"Duplicate transition probability {edge} in {path}")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid transition probability for {edge} in {path}")
        transition_values[edge] = value

    free_edge_set = set(free_edges)
    off_diagonal_edges = {
        edge for edge in transition_values if edge[0] != edge[1]
    }
    if off_diagonal_edges != free_edge_set:
        missing_edges = sorted(free_edge_set - off_diagonal_edges)
        unexpected_edges = sorted(off_diagonal_edges - free_edge_set)
        raise ValueError(
            f"Transition edges in {path} do not match the configured free edges; "
            f"missing={missing_edges}, unexpected={unexpected_edges}"
        )

    initial_transition_probabilities = [
        [0.0 for _ in range(num_states)] for _ in range(num_states)
    ]
    for source in range(num_states):
        outgoing_probability = sum(
            transition_values[edge]
            for edge in free_edge_set
            if edge[0] == source
        )
        diagonal_edge = (source, source)
        diagonal_probability = transition_values.get(
            diagonal_edge,
            1.0 - outgoing_probability,
        )
        if diagonal_probability < -1e-8 or diagonal_probability > 1.0 + 1e-8:
            raise ValueError(
                f"Transition probabilities for state {source} do not form "
                f"a valid row in {path}"
            )
        row_total = outgoing_probability + diagonal_probability
        if not math.isclose(row_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Transition probabilities for state {source} do not sum to 1 "
                f"in {path}"
            )
        initial_transition_probabilities[source][source] = max(
            0.0,
            diagonal_probability,
        )

    for edge in free_edges:
        initial_transition_probabilities[edge[0]][edge[1]] = transition_values[edge]

    return (
        vectors["birth"],
        vectors["death"],
        vectors["sampling"],
        initial_transition_probabilities,
    )


# ============================================================
# MAIN
# ============================================================


def main(
    normalize_root_likelihood: bool | None = None,
    require_initial_parameters: bool = False,
):

    args = parse_args(require_initial_parameters=require_initial_parameters)
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

        free_edges = read_free_edges(args.transition_edges_file)

        print(f"Loaded {len(free_edges)} " f"free transition edges")

        # ====================================================
        # TRANSITION EPOCHS
        # ====================================================

        transition_times = read_transition_times(args.transition_times_file)
        if transition_times[0] == 0.0 and not args.preserve_zero_transition_time:
            transition_times = transition_times[1:]

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


        dats = [
            get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                local_transition_times,
                num_states,
                verbose=args.verbose,
            )
            for (levels, hierarchy), local_transition_times in zip(
                cut_trees, tree_transition_times
            )
        ]

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
        if args.initial_parameters is not None:
            (
                initial_birth_rates,
                initial_death_rates,
                initial_sampling_probabilities,
                initial_transition_probabilities,
            ) = read_initial_parameters(
                args.initial_parameters,
                num_states,
                free_edges,
            )
            print(f"Initializing parameters from {args.initial_parameters}")

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

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
