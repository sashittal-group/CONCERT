from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Node:
    node_id: int
    time: float
    is_leaf: bool
    observed_state: int


@dataclass(frozen=True)
class Edge:
    parent: int
    child: int
    length: float


@dataclass(frozen=True)
class Transition:
    parent: int
    child: int
    source: int
    target: int
    time: float


@dataclass(frozen=True)
class EdgeSegment:
    parent: int
    child: int
    state: int
    start: float
    end: float


@dataclass(frozen=True)
class Partition:
    root: int
    node_states: dict[int, int]
    segments: tuple[EdgeSegment, ...]
    transition_times: tuple[float, ...]
    births: tuple[int, ...]
    leaves: tuple[int, ...]
    exposure: tuple[float, ...]


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_inputs(
    nodes_path: Path,
    edges_path: Path,
    transitions_path: Path,
) -> tuple[dict[int, Node], list[Edge], list[Transition]]:
    nodes: dict[int, Node] = {}
    for row in _read_csv(nodes_path):
        node_id = int(row["node_id"])
        if node_id in nodes:
            raise ValueError(f"Duplicate node_id {node_id} in {nodes_path}")
        nodes[node_id] = Node(
            node_id=node_id,
            time=float(row["T"]),
            is_leaf=bool(int(row["is_leaf"])),
            observed_state=int(row["state_id"]),
        )

    edges = [
        Edge(
            parent=int(row["parent"]),
            child=int(row["child"]),
            length=float(row["branch_length"]),
        )
        for row in _read_csv(edges_path)
    ]
    transitions = [
        Transition(
            parent=int(row["parent"]),
            child=int(row["child"]),
            source=int(row["s0"]),
            target=int(row["s1"]),
            time=float(row["t"]),
        )
        for row in _read_csv(transitions_path)
    ]
    return nodes, edges, transitions


def build_partition(
    nodes: dict[int, Node],
    edges: list[Edge],
    transitions: list[Transition],
    num_states: int,
) -> Partition:
    if num_states < 2:
        raise ValueError("num_states must be at least 2")
    if not nodes or len(edges) != len(nodes) - 1:
        raise ValueError("Input must describe a non-empty tree with N-1 edges")

    children: dict[int, list[int]] = defaultdict(list)
    parents: dict[int, list[int]] = defaultdict(list)
    edge_by_pair: dict[tuple[int, int], Edge] = {}
    for edge in edges:
        if edge.parent not in nodes or edge.child not in nodes:
            raise ValueError(f"Edge {edge.parent}->{edge.child} references an unknown node")
        pair = (edge.parent, edge.child)
        if pair in edge_by_pair:
            raise ValueError(f"Duplicate edge {edge.parent}->{edge.child}")
        edge_by_pair[pair] = edge
        children[edge.parent].append(edge.child)
        parents[edge.child].append(edge.parent)
        parent_time = nodes[edge.parent].time
        child_time = nodes[edge.child].time
        if parent_time < child_time:
            raise ValueError(
                f"Edge {edge.parent}->{edge.child} must not increase in time"
            )
        if not math.isclose(
            edge.length,
            parent_time - child_time,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Branch length disagrees with node times on "
                f"{edge.parent}->{edge.child}"
            )

    roots = set(nodes) - set(parents)
    if len(roots) != 1 or any(len(parent_list) != 1 for parent_list in parents.values()):
        raise ValueError("Edges do not describe one rooted tree")
    root = roots.pop()
    if len(edges) != len(nodes) - 1:
        raise ValueError("Input tree is disconnected or contains a cycle")
    reachable = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children[current]:
            if child not in reachable:
                reachable.add(child)
                stack.append(child)
    if reachable != set(nodes):
        raise ValueError("Input edges do not connect every node to the root")

    for node_id, node in nodes.items():
        degree = len(children[node_id])
        if node.is_leaf != (degree == 0):
            raise ValueError(f"is_leaf disagrees with tree topology for node {node_id}")
        if node_id == root:
            if degree != 1:
                raise ValueError("Expected the original root to have one stem edge")
        elif degree == 1:
            raise ValueError(
                f"Expected a branching or leaf node below the root; node "
                f"{node_id} has one child"
            )

    transition_by_edge: dict[tuple[int, int], list[Transition]] = defaultdict(list)
    for event in transitions:
        pair = (event.parent, event.child)
        if pair not in edge_by_pair:
            raise ValueError(f"Transition references unknown edge {event.parent}->{event.child}")
        if not (0 <= event.source < num_states and 0 <= event.target < num_states):
            raise ValueError(f"Transition on {event.parent}->{event.child} has invalid state")
        if event.source == event.target:
            raise ValueError("A transition must change state")
        parent_time = nodes[event.parent].time
        child_time = nodes[event.child].time
        if not child_time < event.time < parent_time:
            raise ValueError(
                f"Transition time {event.time} is not inside edge "
                f"{event.parent}->{event.child}"
            )
        transition_by_edge[pair].append(event)

    for pair, events in transition_by_edge.items():
        events.sort(key=lambda event: event.time, reverse=True)
        for earlier, later in zip(events, events[1:]):
            if earlier.target != later.source:
                raise ValueError(
                    f"Discontinuous state sequence on edge {pair[0]}->{pair[1]}"
                )

    components = _UnionFind(nodes)
    for edge in edges:
        if (edge.parent, edge.child) not in transition_by_edge:
            components.union(edge.parent, edge.child)

    component_states: dict[int, int] = {}

    def constrain(node_id: int, state: int, origin: str) -> None:
        if not 0 <= state < num_states:
            raise ValueError(f"{origin} assigns invalid state {state}")
        component = components.find(node_id)
        existing = component_states.get(component)
        if existing is not None and existing != state:
            raise ValueError(
                f"Conflicting states {existing} and {state} in the same "
                f"unchanged-state component ({origin})"
            )
        component_states[component] = state

    for node_id, node in nodes.items():
        if node.is_leaf and node.observed_state >= 0:
            constrain(node_id, node.observed_state, f"leaf {node_id}")

    for pair, events in transition_by_edge.items():
        constrain(pair[0], events[0].source, f"transition on edge {pair}")
        constrain(pair[1], events[-1].target, f"transition on edge {pair}")

    node_states: dict[int, int] = {}
    for node_id in nodes:
        component = components.find(node_id)
        if component not in component_states:
            raise ValueError(
                f"State is undetermined for node {node_id}; provide complete "
                "transition history or labeled descendant leaves"
            )
        node_states[node_id] = component_states[component]

    segments: list[EdgeSegment] = []
    for edge in edges:
        events = transition_by_edge.get((edge.parent, edge.child), [])
        start = nodes[edge.parent].time
        current_state = node_states[edge.parent]
        for event in events:
            if event.source != current_state:
                raise ValueError(
                    f"Transition source conflicts with propagated state on "
                    f"{edge.parent}->{edge.child}"
                )
            segments.append(
                EdgeSegment(
                    parent=edge.parent,
                    child=edge.child,
                    state=current_state,
                    start=start,
                    end=event.time,
                )
            )
            start = event.time
            current_state = event.target
        if current_state != node_states[edge.child]:
            raise ValueError(
                f"Transition history does not reach child state on "
                f"{edge.parent}->{edge.child}"
            )
        segments.append(
            EdgeSegment(
                parent=edge.parent,
                child=edge.child,
                state=current_state,
                start=start,
                end=nodes[edge.child].time,
            )
        )

    births = [0] * num_states
    leaves = [0] * num_states
    exposure = [0.0] * num_states
    for node_id, node in nodes.items():
        if node_id != root and not node.is_leaf:
            births[node_states[node_id]] += 1
        if node.is_leaf:
            leaves[node_states[node_id]] += 1
    for segment in segments:
        exposure[segment.state] += segment.start - segment.end

    return Partition(
        root=root,
        node_states=node_states,
        segments=tuple(segments),
        transition_times=tuple(sorted({event.time for event in transitions})),
        births=tuple(births),
        leaves=tuple(leaves),
        exposure=tuple(exposure),
    )


def _exprel_negative(x: torch.Tensor) -> torch.Tensor:
    small = x.abs() < 1e-4
    safe_x = torch.where(small, torch.ones_like(x), x)
    regular = -torch.expm1(-x) / safe_x
    series = 1.0 - x / 2.0 + x.square() / 6.0 - x.pow(3) / 24.0 + x.pow(4) / 120.0
    return torch.where(small, series, regular)


def _scaled_denominator(
    time: float | torch.Tensor,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: float,
) -> torch.Tensor:
    elapsed = torch.as_tensor(time, dtype=birth.dtype, device=birth.device)
    net_rate = birth - death
    exponent = net_rate * elapsed
    return elapsed * _exprel_negative(exponent) + torch.exp(-exponent) / (
        birth * sampling
    )


def single_state_u(
    time: float,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: float,
) -> torch.Tensor:
    if sampling <= 0.0 or sampling >= 1.0:
        raise ValueError("sampling must be strictly between 0 and 1")
    return 1.0 - 1.0 / (
        birth * _scaled_denominator(time, birth, death, sampling)
    )


def segment_log_likelihood(
    segment: EdgeSegment,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: float,
) -> torch.Tensor:
    net_rate = birth - death
    start_denominator = _scaled_denominator(
        segment.start, birth, death, sampling
    )
    end_denominator = _scaled_denominator(segment.end, birth, death, sampling)
    return (
        -net_rate * (segment.start - segment.end)
        + 2.0 * (torch.log(end_denominator) - torch.log(start_denominator))
    )


def state_log_likelihoods(
    partition: Partition,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling: float,
) -> torch.Tensor:
    if birth_rates.ndim != 1 or death_rates.shape != birth_rates.shape:
        raise ValueError("birth_rates and death_rates must be equal-length vectors")
    if birth_rates.numel() != len(partition.births):
        raise ValueError("Rate vectors must match the number of states")

    state_values = []
    for state in range(birth_rates.numel()):
        value = (
            partition.births[state] * torch.log(birth_rates[state])
            + partition.leaves[state] * math.log(sampling)
        )
        for segment in partition.segments:
            if segment.state == state:
                value = value + segment_log_likelihood(
                    segment,
                    birth_rates[state],
                    death_rates[state],
                    sampling,
                )
        state_values.append(value)
    return torch.stack(state_values)


def _logit(value: float) -> float:
    value = min(max(value, 1e-7), 1.0 - 1e-7)
    return math.log(value) - math.log1p(-value)


def fit_partition(
    partition: Partition,
    sampling: float,
    max_birth_rate: float = 20.0,
    max_death_rate: float = 10.0,
    max_iter: int = 1000,
) -> dict[str, object]:
    if not 0.0 < sampling < 1.0:
        raise ValueError("sampling must be strictly between 0 and 1")
    if max_birth_rate <= max_death_rate or max_death_rate <= 0.0:
        raise ValueError("Require max_birth_rate > max_death_rate > 0")

    num_states = len(partition.births)
    if any(partition.births[s] == 0 or partition.exposure[s] <= 0 for s in range(num_states)):
        raise ValueError("Every state must have at least one birth and positive exposure")

    starts = [
        ([5.0] * num_states, [1.0] * num_states),
        ([2.0] * num_states, [0.2] * num_states),
        ([10.0] * num_states, [3.0] * num_states),
    ]
    best: tuple[float, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    for initial_births, initial_deaths in starts:
        death_fraction = torch.tensor(
            [rate / max_death_rate for rate in initial_deaths],
            dtype=torch.float64,
        )
        birth_fraction = torch.tensor(
            [
                (birth - death) / (max_birth_rate - death)
                for birth, death in zip(initial_births, initial_deaths)
            ],
            dtype=torch.float64,
        )
        raw_death = torch.nn.Parameter(
            torch.tensor([_logit(float(x)) for x in death_fraction])
        )
        raw_birth = torch.nn.Parameter(
            torch.tensor([_logit(float(x)) for x in birth_fraction])
        )
        optimizer = torch.optim.LBFGS(
            [raw_death, raw_birth],
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def rates() -> tuple[torch.Tensor, torch.Tensor]:
            death = max_death_rate * torch.sigmoid(raw_death)
            birth = death + (max_birth_rate - death) * torch.sigmoid(raw_birth)
            return birth, death

        def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            birth, death = rates()
            components = state_log_likelihoods(
                partition,
                birth,
                death,
                sampling,
            )
            root_state = partition.node_states[partition.root]
            root_u = single_state_u(
                max(
                    segment.start
                    for segment in partition.segments
                    if segment.parent == partition.root
                ),
                birth[root_state],
                death[root_state],
                sampling,
            )
            total = components.sum() - torch.log1p(-root_u)
            return total, components, birth, death

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            total, _, _, _ = objective()
            loss = -total
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite objective during optimization")
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            total, components, birth, death = objective()
            candidate = (
                float(total),
                components.detach().clone(),
                birth.detach().clone(),
                death.detach().clone(),
            )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    total, components, birth, death = best
    root_state = partition.node_states[partition.root]
    root_time = max(
        segment.start
        for segment in partition.segments
        if segment.parent == partition.root
    )
    root_u = single_state_u(
        root_time,
        birth[root_state],
        death[root_state],
        sampling,
    )
    u_at_transition_times = {
        f"{time:.15g}": [
            float(single_state_u(time, birth[s], death[s], sampling))
            for s in range(num_states)
        ]
        for time in partition.transition_times
    }

    return {
        "log_likelihood": total,
        "state_log_likelihoods": components.tolist(),
        "birth_rates": birth.tolist(),
        "death_rates": death.tolist(),
        "sampling_probabilities": [sampling] * num_states,
        "root_state": root_state,
        "root_time": root_time,
        "root_U": float(root_u),
        "root_normalization": float(-torch.log1p(-root_u)),
        "U_at_transition_times": u_at_transition_times,
    }


def run_inference(
    nodes_path: Path,
    edges_path: Path,
    transitions_path: Path,
    output_path: Path,
    num_states: int = 3,
    sampling: float = 0.05,
) -> dict[str, object]:
    nodes, edges, transitions = read_inputs(nodes_path, edges_path, transitions_path)
    partition = build_partition(nodes, edges, transitions, num_states)
    fit = fit_partition(partition, sampling)
    result: dict[str, object] = {
        "method": "single-state likelihood on known per-edge state segments",
        "assumptions": [
            "Listed transitions are the complete state history; unlisted edges retain state.",
            "Transition times and directions are conditioned on; transition probabilities are not included.",
            "Sampling probabilities are fixed to the requested common value.",
            "Root normalization is applied once at the original tree root.",
        ],
        "state_names": ["NC", "PT", "LN"] if num_states == 3 else [
            f"state_{state}" for state in range(num_states)
        ],
        "root_node": partition.root,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_transitions": len(transitions),
        "birth_counts": list(partition.births),
        "leaf_counts": list(partition.leaves),
        "lineage_exposure": list(partition.exposure),
        "node_states": {
            str(node_id): state for node_id, state in partition.node_states.items()
        },
        **fit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit state-specific single-state birth-death likelihoods using "
            "known per-edge transition histories."
        )
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/sgootr/inference/nodes.csv"),
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/sgootr/inference/edges.csv"),
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=Path("data/sgootr/inference/transitions_0_adjusted.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sgootr/inference/inference_single_state.json"),
    )
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--sampling", type=float, default=0.05)
    args = parser.parse_args()

    result = run_inference(
        args.nodes,
        args.edges,
        args.transitions,
        args.output,
        num_states=args.num_states,
        sampling=args.sampling,
    )
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
