from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import math
import re

import torch

from src.concert.tree_csv import load_edges, load_nodes
from src.concert.tree_utils import (
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
    tree_sampling_scope: str | None = None
    tree_transition_times: list[list[float]] | None = None

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
    tree_sampling_probabilities: torch.Tensor | None = None
    tree_sampling_scope: str | None = None
    tree_transition_probabilities: torch.Tensor | None = None
    tree_transition_times: list[list[float]] | None = None


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


def _parse_tree_sampling_probabilities(
    lines: list[str],
    num_states: int,
    statewise_mean: torch.Tensor,
    path: Path,
) -> tuple[torch.Tensor | None, str | None]:
    header = "sampling_by_tree:"
    if header not in lines:
        return None, None
    scope_lines = [
        line.partition(":")[2].strip()
        for line in lines
        if line.startswith("sampling_scope:")
    ]
    if len(scope_lines) != 1:
        raise ValueError(f"Expected one sampling_scope in {path}")
    sampling_scope = scope_lines[0]
    if sampling_scope not in {
        "per_tree_shared_across_states",
        "per_tree_by_state",
    }:
        raise ValueError(f"Invalid per-tree sampling scope in {path}: {sampling_scope}")
    start = lines.index(header) + 1
    values: dict[int, list[float]] = {}
    for line in lines[start:]:
        if line == "transitions:":
            break
        match = re.fullmatch(r"tree_(\d+):\s*(.*)", line)
        if match is None:
            if line.strip():
                raise ValueError(f"Invalid per-tree sampling row in {path}: {line}")
            continue
        tree_index = int(match.group(1))
        if tree_index in values:
            raise ValueError(f"Duplicate tree sampling row {tree_index} in {path}")
        try:
            values[tree_index] = [float(value) for value in match.group(2).split()]
        except ValueError as error:
            raise ValueError(
                f"Invalid per-tree sampling values in {path}: {line}"
            ) from error
    if not values or sorted(values) != list(range(len(values))):
        raise ValueError(f"Per-tree sampling indices in {path} must start at zero.")
    expected_width = (
        1
        if sampling_scope == "per_tree_shared_across_states"
        else num_states
    )
    if any(len(tree_values) != expected_width for tree_values in values.values()):
        raise ValueError(
            f"Expected {expected_width} sampling probabilities per tree in {path}"
        )
    probabilities = torch.tensor(
        [values[index] for index in range(len(values))],
        dtype=torch.float64,
    )
    if not torch.all(torch.isfinite(probabilities)) or torch.any(
        (probabilities <= 0.0) | (probabilities >= 1.0)
    ):
        raise ValueError(f"Per-tree sampling probabilities in {path} are invalid.")
    mean_probabilities = (
        probabilities.mean().expand(num_states)
        if sampling_scope == "per_tree_shared_across_states"
        else probabilities.mean(dim=0)
    )
    if not torch.allclose(mean_probabilities, statewise_mean, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"Per-tree sampling mean does not match the sampling vector in {path}"
        )
    return (
        probabilities.flatten()
        if sampling_scope == "per_tree_shared_across_states"
        else probabilities,
        sampling_scope,
    )


def _parse_transition_matrix(lines: list[str], num_states: int, path: Path) -> torch.Tensor:
    probabilities = torch.zeros((num_states, num_states), dtype=torch.float64)
    diagonal_entries = set()
    in_transitions = False
    for line in lines:
        line = line.strip()
        if line == "transitions:":
            in_transitions = True
            continue
        if line == "transition_times:":
            in_transitions = False
            continue
        if line.startswith("transition_scope:"):
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
            if source_index == target_index:
                diagonal_entries.add(source_index)

    for state in range(num_states):
        if state not in diagonal_entries:
            off_diagonal_sum = float(probabilities[state].sum())
            if off_diagonal_sum > 1.0:
                raise ValueError(f"Transition matrix in {path} is not row-stochastic")
            probabilities[state, state] = 1.0 - off_diagonal_sum

    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(num_states, dtype=torch.float64),
        atol=1e-6,
    ):
        raise ValueError(f"Transition matrix in {path} is not row-stochastic")
    return probabilities


def _parse_tree_transition_probabilities(
    lines: list[str],
    num_states: int,
    statewise_mean: torch.Tensor,
    path: Path,
) -> torch.Tensor | None:
    header = "transition_probabilities_by_tree:"
    if header not in lines:
        return None
    values: dict[int, torch.Tensor] = {}
    tree_index: int | None = None
    for line in lines[lines.index(header) + 1 :]:
        if line in {"transition_times:", "transition_times_by_tree:"}:
            break
        tree_match = re.fullmatch(r"tree_(\d+):", line)
        if tree_match:
            tree_index = int(tree_match.group(1))
            if tree_index in values:
                raise ValueError(f"Duplicate tree transition matrix in {path}")
            values[tree_index] = torch.zeros(
                (num_states, num_states),
                dtype=torch.float64,
            )
            continue
        edge_match = re.fullmatch(r"(\d+)->(\d+)\s+([0-9.eE+-]+)", line)
        if edge_match is None:
            if line.strip():
                raise ValueError(f"Invalid per-tree transition row in {path}: {line}")
            continue
        if tree_index is None:
            raise ValueError(f"Transition edge before tree label in {path}: {line}")
        source, target, raw_probability = edge_match.groups()
        source_index, target_index = int(source), int(target)
        probability = float(raw_probability)
        if (
            source_index >= num_states
            or target_index >= num_states
            or not math.isfinite(probability)
            or probability < 0.0
        ):
            raise ValueError(f"Invalid per-tree transition probability in {path}")
        values[tree_index][source_index, target_index] = probability
    if not values or sorted(values) != list(range(len(values))):
        raise ValueError(f"Per-tree transition indices in {path} must start at zero")
    probabilities = torch.stack([values[index] for index in range(len(values))])
    if not torch.allclose(
        probabilities.sum(dim=2),
        torch.ones((len(values), num_states), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(f"Per-tree transition matrices in {path} must be row-stochastic")
    if not torch.allclose(
        probabilities.mean(dim=0),
        statewise_mean,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"Per-tree transition mean does not match the transition matrix in {path}"
        )
    return probabilities


def _parse_tree_transition_times(
    lines: list[str],
    path: Path,
) -> list[list[float]] | None:
    header = "transition_times_by_tree:"
    if header not in lines:
        return None
    values: dict[int, list[float]] = {}
    for line in lines[lines.index(header) + 1 :]:
        if not line.strip():
            continue
        match = re.fullmatch(r"tree_(\d+):\s*(.*)", line)
        if match is None:
            raise ValueError(f"Invalid per-tree transition times in {path}: {line}")
        tree_index = int(match.group(1))
        if tree_index in values:
            raise ValueError(f"Duplicate tree transition times in {path}")
        try:
            tree_times = [float(value) for value in match.group(2).split()]
        except ValueError as error:
            raise ValueError(f"Invalid per-tree transition times in {path}") from error
        if (
            not tree_times
            or any(not math.isfinite(value) or value <= 0.0 for value in tree_times)
            or any(right <= left for left, right in zip(tree_times, tree_times[1:]))
        ):
            raise ValueError(f"Per-tree transition times in {path} must be positive and sorted")
        values[tree_index] = tree_times
    if not values or sorted(values) != list(range(len(values))):
        raise ValueError(f"Per-tree transition-time indices in {path} must start at zero")
    return [values[index] for index in range(len(values))]


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
    tree_sampling_probabilities, tree_sampling_scope = _parse_tree_sampling_probabilities(
        lines,
        num_states,
        sampling_probabilities,
        inference_path,
    )
    tree_transition_probabilities = _parse_tree_transition_probabilities(
        lines,
        num_states,
        _parse_transition_matrix(lines, num_states, inference_path),
        inference_path,
    )
    tree_transition_times = _parse_tree_transition_times(lines, inference_path)
    if (tree_transition_probabilities is None) != (tree_transition_times is None):
        raise ValueError(
            f"Per-tree transition matrices and times must both be present in "
            f"{inference_path}"
        )
    if (
        tree_transition_probabilities is not None
        and len(tree_transition_probabilities) != len(tree_transition_times or [])
    ):
        raise ValueError(
            f"Per-tree transition matrix/time counts differ in {inference_path}"
        )

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
        tree_sampling_probabilities=tree_sampling_probabilities,
        tree_sampling_scope=tree_sampling_scope,
        tree_transition_probabilities=tree_transition_probabilities,
        tree_transition_times=tree_transition_times,
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
        sampling_probabilities=(
            parameters.tree_sampling_probabilities
            if parameters.tree_sampling_probabilities is not None
            and parameters.tree_sampling_probabilities.ndim == 2
            else parameters.sampling_probabilities
        ),
        transition_probabilities=parameters.transition_probabilities,
        transition_times=parameters.transition_times,
        sensitivity_inputs=sensitivity_inputs,
        tree_sampling_scope=parameters.tree_sampling_scope,
        tree_transition_times=parameters.tree_transition_times,
    )
