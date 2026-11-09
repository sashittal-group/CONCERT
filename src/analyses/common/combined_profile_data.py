from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re

from src.concert.tree_csv import load_edges, load_nodes
from src.concert.infer_given_transitions import (
    find_tree_file_pairs,
    load_per_edge_transition_data,
)
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


def load_combined_dataset_context(
    input_dir: Path,
    inference_path: Path,
    state_names: tuple[str, ...],
    *,
    use_per_edge_transition_data: bool = True,
    allow_stem_root: bool = False,
) -> DatasetContext:
    parameters = load_inference_parameters(inference_path, state_names)
    tree_file_pairs = find_tree_file_pairs(input_dir)
    if not tree_file_pairs:
        raise ValueError(
            f"Expected at least one tree pair in {input_dir}; found none."
        )
    if (
        parameters.tree_transition_probabilities is not None
        and parameters.tree_transition_probabilities.shape[0] != len(tree_file_pairs)
    ):
        raise ValueError(
            "Per-tree transition matrix count does not match the combined tree count"
        )
    if (
        parameters.tree_transition_times is not None
        and len(parameters.tree_transition_times) != len(tree_file_pairs)
    ):
        raise ValueError(
            "Per-tree transition-time count does not match the combined tree count"
        )

    sensitivity_inputs = []
    for tree_position, (node_path, edge_path) in enumerate(tree_file_pairs):
        match = re.search(r"nodes_(\d+)", node_path.stem, flags=re.IGNORECASE)
        tree_index = int(match.group(1)) if match else tree_position
        transition_path = input_dir / f"transition_times_{tree_index}.csv"
        with redirect_stdout(StringIO()):
            transition_data = (
                load_per_edge_transition_data(transition_path)
                if use_per_edge_transition_data
                else None
            )
            nodes = load_nodes(str(node_path))
            edges = load_edges(str(edge_path))
            tree = prepare_nx_tree(nodes, edges)

        root = tree.graph["root"]
        if tree.out_degree(root) != 2 and not (
            allow_stem_root and tree.out_degree(root) == 1
        ):
            raise ValueError(
                f"{node_path} must have a binary crown root"
                f"{' or a single stem root' if allow_stem_root else ''}."
            )
        invalid_leaf_states = [
            node.state
            for node in nodes
            if node.is_leaf and not 0 <= node.state < len(state_names)
        ]
        if invalid_leaf_states:
            raise ValueError(
                f"Invalid leaf states in {node_path}: "
                f"{sorted(set(invalid_leaf_states))}"
            )

        root_time = float(tree.nodes[root]["T"])
        candidate_times = (
            parameters.tree_transition_times[tree_position]
            if parameters.tree_transition_times is not None
            else parameters.transition_times
        )
        local_transition_times = [
            time for time in candidate_times if time < root_time
        ]
        data_structure_times = (
            local_transition_times
            if use_per_edge_transition_data
            else parameters.transition_times
        )
        levels, hierarchy = cut_tree(tree, local_transition_times)
        with redirect_stdout(StringIO()):
            datastructure = get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                data_structure_times,
                len(state_names),
                per_edge_transition_data=transition_data,
            )
        sensitivity_inputs.append(
            SubtreeLikelihoodInput(
                root_time,
                data_structure_times,
                datastructure,
            )
        )

    return DatasetContext(
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=(
            (
                parameters.tree_sampling_probabilities[:, None]
                .expand(-1, len(state_names))
                .clone()
                if parameters.tree_sampling_probabilities.ndim == 1
                else parameters.tree_sampling_probabilities
            )
            if parameters.tree_sampling_probabilities is not None
            else parameters.sampling_probabilities
        ),
        transition_probabilities=(
            parameters.tree_transition_probabilities
            if parameters.tree_transition_probabilities is not None
            else parameters.transition_probabilities
        ),
        transition_times=parameters.transition_times,
        sensitivity_inputs=sensitivity_inputs,
        tree_sampling_scope=parameters.tree_sampling_scope,
        tree_transition_times=parameters.tree_transition_times,
    )
