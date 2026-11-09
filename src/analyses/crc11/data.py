from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.concert.tree_csv import load_edges, load_nodes
from src.concert.infer_given_transitions import load_per_edge_transition_data
from src.analyses.common.profile_data import (
    DatasetContext,
    SubtreeLikelihoodInput,
    load_inference_parameters,
)
from src.concert.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


def load_dataset_context(
    dataset_root: Path,
    state_names: tuple[str, ...],
    inference_path: Path | None = None,
) -> DatasetContext:
    parameters = load_inference_parameters(
        inference_path or dataset_root / "inference_result.txt",
        state_names,
    )
    with redirect_stdout(StringIO()):
        nodes = load_nodes(str(dataset_root / "nodes.csv"))
        edges = load_edges(str(dataset_root / "edges.csv"))
    tree = prepare_nx_tree(nodes, edges)
    root = tree.graph["root"]
    if tree.out_degree(root) != 2:
        raise ValueError(
            f"{dataset_root} must have a stem-free binary crown root."
        )
    num_states = len(state_names)
    invalid_leaf_states = [
        node.state
        for node in nodes
        if node.is_leaf and not 0 <= node.state < num_states
    ]
    if invalid_leaf_states:
        raise ValueError(
            f"Invalid leaf states in {dataset_root}: {sorted(set(invalid_leaf_states))}"
        )

    root_time = float(tree.nodes[root]["T"])
    transition_times = [
        transition_time
        for transition_time in parameters.transition_times
        if transition_time < root_time
    ]
    per_edge_transition_data = load_per_edge_transition_data(
        dataset_root / "transition_times.csv"
    )
    levels, hierarchy = cut_tree(tree, transition_times)
    with redirect_stdout(StringIO()):
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            transition_times,
            num_states,
            per_edge_transition_data=per_edge_transition_data,
        )

    return DatasetContext(
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=parameters.sampling_probabilities,
        transition_probabilities=parameters.transition_probabilities,
        transition_times=transition_times,
        sensitivity_inputs=[
            SubtreeLikelihoodInput(root_time, transition_times, datastructure)
        ],
    )
