# main.py

from __future__ import annotations

import argparse
from platform import node
import time
from pathlib import Path
import pandas as pd

import os

# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"

import torch

from src.dataprocessor import load_nodes, load_edges, load_transition_times
from src.dtos import Edge, Node
from src.tree_utils import prepare_tree

from src.mle_given_transitions import (
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


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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



def prepare_data(nodes: list[Node], edges: list[Edge], transition_times_df: pd.DataFrame, num_states: int):

    internal_node_state_counts = [0] * num_states
    leaf_state_counts = [0] * num_states

    node_dict = {}

    for node in nodes:
        node_dict[node.id] = node

        if node.is_leaf:
            leaf_state_counts[node.state] += 1
        else:
            internal_node_state_counts[node.state] += 1


    state_egdes = []
    for _ in range(num_states):
        state_egdes.append([])

    for edge in edges:
        transition_filt = transition_times_df[
            (transition_times_df['parent'] == edge.u) & 
            (transition_times_df['child'] == edge.v)
        ]

        start_time = node_dict[edge.u].T
        end_time = node_dict[edge.v].T
        start_state = node_dict[edge.u].state
        end_state = node_dict[edge.v].state

        if not transition_filt.empty:
            transition_times = [
                (d.parent, d.child, d.s0, d.s1, d.t) for d in transition_filt.itertuples()
            ]
            transition_times.sort(key=lambda x: x[4], reverse=True)

            for p, c, s0, s1, t in transition_times:
                state_egdes[start_state].append((start_time, t))
                start_time = t
                start_state = s1

        state_egdes[start_state].append((start_time, end_time))

    return internal_node_state_counts, leaf_state_counts, state_egdes



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

    normalize_root_likelihood = args.normalize_root_likelihood

    print(f"dir={args.directory}")

    print(f"num_states={num_states}")

    print(f"num_trees={num_trees}")

    print(f"output={output_file}")

    print(f"logfile={log_file}")

    print(f"prior=" f"{'true' if args.prior else 'false'}")

    print(f"device={device}")

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
            transition_times_path = directory / f"transitions_{i}.csv"

            loaded_nodes = load_nodes(str(node_path))

            loaded_edges = load_edges(str(edge_path))

            df_transition_times = load_transition_times(str(transition_times_path))

            if len(loaded_nodes) == 0:
                raise RuntimeError(f"No nodes loaded from " f"{node_path}")

            if len(loaded_edges) == 0:
                raise RuntimeError(f"No edges loaded from " f"{edge_path}")

            if len(df_transition_times) == 0:
                raise RuntimeError(f"No transition times loaded from " f"{transition_times_path}")

            internal_node_state_counts, leaf_state_counts, state_egdes = prepare_data(
                loaded_nodes,
                loaded_edges,
                df_transition_times,
                num_states
            )

            trees.append(
                (internal_node_state_counts, leaf_state_counts, state_egdes)
            )

            print(
                f"[timing] tree {i} preparation: "
                f"{time.perf_counter() - tree_start:.3f}s "
                f"({len(loaded_nodes)} nodes, {len(loaded_edges)} edges)",
                flush=True,
            )

        print(f"LOADED {len(trees)} TREES")

        initial_birth_rates = None
        initial_death_rates = None
        initial_sampling_probabilities = None
        
        # ====================================================
        # FIXED PARAMETERS
        # ====================================================

        given_birth_rates = None
        # given_death_rates = [0.2, 0.5]

        # ====================================================
        # OPTIMIZATION
        # ====================================================

        result = optimize_parameters(
            trees=trees,
            num_states=num_states,
            logfile=log_file,
            initial_birth_rates=(initial_birth_rates),
            initial_death_rates=(initial_death_rates),
            initial_sampling_probabilities=(initial_sampling_probabilities),
            # given_death_rates=(given_death_rates),
            # given_birth_rates=(given_birth_rates),
            # given_sampling_probabilities=(given_sampling_probabilities),
            optimizer_name=(args.optimizer),
            learning_rate=(args.learning_rate),
            max_steps=(args.max_steps),
            patience=(args.patience),
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
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
