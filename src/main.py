# main.py

from __future__ import annotations

import argparse
import time
from pathlib import Path


import os

# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"

import torch

from src.dataprocessor import load_nodes, load_edges
from src.tree_utils import prepare_tree

from src.mle_closed_form import (
    optimize_parameters,
)
# from src.closed_form_coord import (
#     calculate_linear_prior_for_transition_times,
# )


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
        help="Directory containing nodes_0.csv, edges_0.csv, etc.",
    )

    parser.add_argument(
        "--num-states",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-trees",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--transition-edges-file",
        type=str,
        required=True,
    )

    # parser.add_argument(
    #     "--transition-times-file",
    #     type=str,
    #     default="",
    # )

    parser.add_argument(
        "--num-transition-epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--zero-death",
        action="store_true",
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
        "--device",
        type=str,
        default=None,
        help="cuda, mps, or cpu. Default: automatically detect.",
    )

    return parser.parse_args()


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


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    num_states = args.num_states
    num_trees = args.num_trees

    output_file = args.output
    log_file = args.log

    device = get_device(args.device)

    print(f"dir={args.directory}")

    print(f"num_states={num_states}")

    print(f"num_trees={num_trees}")

    print(f"output={output_file}")

    print(f"logfile={log_file}")

    # print(
    #     f"num_transition_epochs="
    #     f"{num_transition_epochs}"
    # )

    print(f"prior=" f"{'true' if args.prior else 'false'}")

    print(f"device={device}")

    if args.transition_edges_file:
        print("transition_edges_file=" f"{args.transition_edges_file}")

    # if args.transition_times_file:
    #     print(
    #         "transition_times_file="
    #         f"{args.transition_times_file}"
    #     )

    print("STARTED ...")

    start = time.perf_counter()

    try:

        # ====================================================
        # LOAD TREES
        # ====================================================

        directory = Path(args.directory)

        trees = []

        for i in range(num_trees):

            tree_start = time.perf_counter()

            node_path = directory / f"nodes_{i}.csv"

            edge_path = directory / f"edges_{i}.csv"

            loaded_nodes = load_nodes(str(node_path))

            loaded_edges = load_edges(str(edge_path))

            if len(loaded_nodes) == 0:
                raise RuntimeError(f"No nodes loaded from " f"{node_path}")

            if len(loaded_edges) == 0:
                raise RuntimeError(f"No edges loaded from " f"{edge_path}")

            tree = prepare_tree(
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

        # Same as:
        #
        # transition_epochs.push_back(
        #     {0.0, {}}
        # );
        #
        # transition_epochs.push_back(
        #     {0.5 - 1e-6, free_edges}
        # );

        transition_times = [0.0, 0.5 - 1e-6]
        
        # ====================================================
        # PRIOR
        # ====================================================

        if args.prior:

            factor = 200.0

            print(f"Using linear prior for " f"transition times with factor {factor}")

        else:

            print(f"No prior for transition times")

            factor = 0.0

        # def prior_fn(epochs):

        #     return calculate_linear_prior_for_transition_times(
        #         epochs,
        #         factor,
        #     )

        # Do not use a prior function when --prior is false.
        # actual_prior_fn = prior_fn if args.prior else None

        # ====================================================
        # INITIAL PARAMETERS
        # ====================================================

        initial_birth_rates = [8, 7]

        initial_death_rates = [0.5, 0.2]

        initial_sampling_probabilities = [0.5, 0.6]

        # Same as the C++:
        #
        # const vector<vector<long double>>
        # initial_transition_probabilities = {
        #     {0.5, 0.5},
        #     {0.0, 1.0}
        # };

        initial_transition_probabilities = [
            [0.5, 0.5],
            [0.0, 1.0],
        ]

        initial_transition_times = None

        # ====================================================
        # FIXED PARAMETERS
        # ====================================================

        given_birth_rates = None

        # Same as:
        #
        # given_sampling_probabilities = {0.5, 0.6};

        given_sampling_probabilities = [
            0.5,
            0.6,
        ]

        # ====================================================
        # VALIDATION
        # ====================================================

        if num_states != 2:

            raise ValueError(
                "The hard-coded initial transition "
                "probabilities and given sampling "
                "probabilities in this main file are "
                "defined for num_states=2. "
                "Provide generalized values before using "
                "--num-states other than 2."
            )

        if (
            initial_transition_probabilities
            and len(initial_transition_probabilities) != num_states
        ):
            raise ValueError(
                "initial_transition_probabilities " "has incorrect number of rows"
            )

        if (
            given_sampling_probabilities
            and len(given_sampling_probabilities) != num_states
        ):
            raise ValueError("given_sampling_probabilities " "has incorrect length")

        # ====================================================
        # OPTIMIZATION
        # ====================================================

        result = optimize_parameters(
            trees=trees,
            num_states=num_states,
            given_transition_times=(transition_times),
            free_edges=free_edges,
            logfile=log_file,
            initial_birth_rates=(initial_birth_rates),
            initial_death_rates=(initial_death_rates),
            initial_sampling_probabilities=(initial_sampling_probabilities),
            initial_transition_probabilities=(initial_transition_probabilities),
            # given_birth_rates=(given_birth_rates),
            # given_sampling_probabilities=(given_sampling_probabilities),
            optimizer_name=(args.optimizer),
            learning_rate=(args.learning_rate),
            max_steps=(args.max_steps),
            device=device,
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
