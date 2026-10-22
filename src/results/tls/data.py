from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import torch

from src.dataprocessor import load_edges, load_nodes
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


@dataclass(frozen=True)
class SubtreeLikelihoodInput:
    root_time: float
    transition_times: list[float]
    datastructure: object


@dataclass(frozen=True)
class DatasetContext:
    birth_rates: torch.Tensor
    death_rates: torch.Tensor
    sampling_probabilities: torch.Tensor
    transition_probabilities: torch.Tensor
    transition_times: list[float]
    sensitivity_inputs: list[SubtreeLikelihoodInput]

    @property
    def root_times(self) -> list[float]:
        return [subtree.root_time for subtree in self.sensitivity_inputs]


@dataclass(frozen=True)
class InferenceParameters:
    birth_rates: torch.Tensor
    death_rates: torch.Tensor
    sampling_probabilities: torch.Tensor
    transition_probabilities: torch.Tensor
    transition_times: list[float]


def _parse_parameter_vector(lines: list[str], name: str, path: Path) -> list[float]:
    prefix = f"{name}:"
    matching_lines = [line for line in lines if line.startswith(prefix)]
    if len(matching_lines) != 1:
        raise ValueError(
            f"Expected one {name!r} vector in {path}; found {len(matching_lines)}"
        )
    try:
        return [float(value) for value in matching_lines[0][len(prefix):].split()]
    except ValueError as error:
        raise ValueError(f"Invalid {name!r} vector in {path}") from error


def _parse_transition_matrix(lines: list[str], num_states: int, path: Path) -> torch.Tensor:
    probabilities = torch.zeros((num_states, num_states), dtype=torch.float64)
    in_transitions = False
    for line in lines:
        line = line.strip()
        if line == "transitions:":
            in_transitions = True
            continue
        if line == "transition_times:":
            in_transitions = False
            continue
        if not in_transitions:
            continue
        match = re.fullmatch(r"(\d+)->(\d+)\s+([0-9.eE+-]+)", line)
        if match:
            source, target, probability = match.groups()
            source_index, target_index = int(source), int(target)
            if source_index >= num_states or target_index >= num_states:
                raise ValueError(f"Transition index outside matrix dimensions in {path}")
            probabilities[source_index, target_index] = float(probability)

    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(num_states, dtype=torch.float64),
        atol=1e-6,
    ):
        raise ValueError(f"Transition matrix in {path} is not row-stochastic")
    return probabilities


def _parse_transition_times(lines: list[str], path: Path) -> list[float]:
    transition_times: list[float] = []
    in_transition_times = False
    for line in lines:
        line = line.strip()
        if line == "transition_times:":
            in_transition_times = True
            continue
        if in_transition_times and line:
            fields = line.split()
            if len(fields) == 2 and fields[0].isdigit():
                try:
                    transition_times.append(float(fields[1]))
                except ValueError as error:
                    raise ValueError(f"Invalid transition time in {path}: {line}") from error
    if not transition_times:
        raise ValueError(f"No transition times found in {path}")
    return transition_times


def _load_subtree_inputs(
    dataset_root: Path,
    transition_times: list[float],
    num_states: int,
) -> list[SubtreeLikelihoodInput]:
    subtree_root = dataset_root / "subtrees_kept_states"
    node_files = sorted(subtree_root.glob("tree_*/subtree_*/nodes.csv"))
    if not node_files:
        raise FileNotFoundError(f"No exported subtree nodes found under {subtree_root}")

    inputs = []
    for node_file in node_files:
        edge_file = node_file.with_name("edges.csv")
        if not edge_file.is_file():
            raise FileNotFoundError(f"Missing edge file for {node_file}")
        with redirect_stdout(StringIO()):
            nodes = load_nodes(str(node_file))
            edges = load_edges(str(edge_file))
        tree = prepare_nx_tree(nodes, edges)
        invalid_leaf_states = [
            node.state
            for node in nodes
            if node.is_leaf and not 0 <= node.state < num_states
        ]
        if invalid_leaf_states:
            raise ValueError(
                f"Invalid leaf states in {node_file}: {sorted(set(invalid_leaf_states))}"
            )

        root = tree.graph["root"]
        root_time = float(tree.nodes[root]["T"])
        applicable_times = [time for time in transition_times if time < root_time]
        levels, hierarchy = cut_tree(tree, applicable_times)
        with redirect_stdout(StringIO()):
            datastructure = get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                applicable_times,
                num_states,
            )
        inputs.append(
            SubtreeLikelihoodInput(root_time, applicable_times, datastructure)
        )
    return inputs


def load_inference_parameters(
    inference_path: Path,
    state_names: tuple[str, ...],
) -> InferenceParameters:
    lines = inference_path.read_text().splitlines()
    birth_rates = torch.tensor(
        _parse_parameter_vector(lines, "birth", inference_path), dtype=torch.float64
    )
    death_rates = torch.tensor(
        _parse_parameter_vector(lines, "death", inference_path), dtype=torch.float64
    )
    sampling_probabilities = torch.tensor(
        _parse_parameter_vector(lines, "sampling", inference_path),
        dtype=torch.float64,
    )
    num_states = len(birth_rates)
    if num_states != len(state_names):
        raise ValueError(
            f"Expected {len(state_names)} states in {inference_path.parent.name}; "
            f"found {num_states}"
        )
    if len(death_rates) != num_states or len(sampling_probabilities) != num_states:
        raise ValueError(f"Inconsistent inferred parameter lengths in {inference_path}")

    return InferenceParameters(
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_probabilities=sampling_probabilities,
        transition_probabilities=_parse_transition_matrix(
            lines,
            num_states,
            inference_path,
        ),
        transition_times=_parse_transition_times(lines, inference_path),
    )


def load_dataset_context(
    dataset_root: Path,
    state_names: tuple[str, ...],
    inference_path: Path | None = None,
) -> DatasetContext:
    if inference_path is None:
        inference_path = dataset_root / "inference.out"
    parameters = load_inference_parameters(inference_path, state_names)
    sensitivity_inputs = _load_subtree_inputs(
        dataset_root,
        parameters.transition_times,
        len(parameters.birth_rates),
    )
    return DatasetContext(
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=parameters.sampling_probabilities,
        transition_probabilities=parameters.transition_probabilities,
        transition_times=parameters.transition_times,
        sensitivity_inputs=sensitivity_inputs,
    )
