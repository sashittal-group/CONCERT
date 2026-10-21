from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class TreeData:
    root: int
    root_time: float
    births: int
    leaves: int
    edges: tuple[tuple[float, float], ...]
    num_nodes: int
    zero_length_edges: int
    multifurcating_nodes: int


def load_tree(nodes_path: Path, edges_path: Path) -> TreeData:
    nodes: dict[int, tuple[float, bool]] = {}
    with nodes_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            node_id = int(row["node_id"])
            if node_id in nodes:
                raise ValueError(f"Duplicate node {node_id}")
            nodes[node_id] = (float(row["T"]), bool(int(row["is_leaf"])))

    edges: list[tuple[int, int]] = []
    children: dict[int, int] = {node_id: 0 for node_id in nodes}
    parents: dict[int, int] = {node_id: 0 for node_id in nodes}
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    zero_length_edges = 0
    with edges_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parent, child = int(row["parent"]), int(row["child"])
            if parent not in nodes or child not in nodes:
                raise ValueError(f"Unknown node on edge {parent}->{child}")
            parent_time, child_time = nodes[parent][0], nodes[child][0]
            length = float(row["branch_length"])
            if parent_time < child_time:
                raise ValueError(f"Time increases on edge {parent}->{child}")
            if not math.isclose(
                length,
                parent_time - child_time,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Branch length disagrees with times on {parent}->{child}")
            if length == 0.0:
                zero_length_edges += 1
            edges.append((parent, child))
            children[parent] += 1
            parents[child] += 1
            adjacency[parent].append(child)

    roots = [node_id for node_id, count in parents.items() if count == 0]
    if len(roots) != 1 or len(edges) != len(nodes) - 1:
        raise ValueError("Input must be one rooted tree with N-1 edges")
    root = roots[0]
    if children[root] != 1:
        raise ValueError("Expected the original tree root to have one stem edge")
    if any(count != 1 for node_id, count in parents.items() if node_id != root):
        raise ValueError("Every non-root node must have exactly one parent")

    reachable = {root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in adjacency[parent]:
            if child not in reachable:
                reachable.add(child)
                stack.append(child)
    if reachable != set(nodes):
        raise ValueError("Input tree has nodes disconnected from its root")

    births = 0
    leaves = 0
    multifurcating_nodes = 0
    for node_id, (_, is_leaf) in nodes.items():
        degree = children[node_id]
        if is_leaf != (degree == 0):
            raise ValueError(f"is_leaf disagrees with topology at node {node_id}")
        if node_id != root and degree > 0:
            births += degree - 1
        if node_id != root and degree > 2:
            multifurcating_nodes += 1
        if is_leaf:
            leaves += 1

    if births != leaves - 1:
        raise ValueError("Binary birth-event count does not match sampled tree size")

    edge_times = tuple(
        (nodes[parent][0], nodes[child][0]) for parent, child in edges
    )
    return TreeData(
        root=root,
        root_time=nodes[root][0],
        births=births,
        leaves=leaves,
        edges=edge_times,
        num_nodes=len(nodes),
        zero_length_edges=zero_length_edges,
        multifurcating_nodes=multifurcating_nodes,
    )


def _logit(value: float) -> float:
    value = min(max(value, 1e-8), 1.0 - 1e-8)
    return math.log(value) - math.log1p(-value)


def _exprel_negative(value: torch.Tensor) -> torch.Tensor:
    small = value.abs() < 1e-4
    safe_value = torch.where(small, torch.ones_like(value), value)
    regular = -torch.expm1(-value) / safe_value
    series = (
        1.0
        - value / 2.0
        + value.square() / 6.0
        - value.pow(3) / 24.0
        + value.pow(4) / 120.0
    )
    return torch.where(small, series, regular)


def _scaled_denominator(
    time: float,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
) -> torch.Tensor:
    elapsed = torch.as_tensor(time, dtype=birth.dtype, device=birth.device)
    net_rate = birth - death
    scaled_time = net_rate * elapsed
    return elapsed * _exprel_negative(scaled_time) + torch.exp(
        -scaled_time
    ) / (birth * sampling)


def calculate_log_likelihood(
    tree: TreeData,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
    normalize_root: bool = True,
) -> torch.Tensor:
    if torch.any(birth <= death) or torch.any(birth <= 0.0):
        raise ValueError("Require birth > death >= 0")
    if torch.any(sampling <= 0.0) or torch.any(sampling >= 1.0):
        raise ValueError("Sampling probability must be strictly between 0 and 1")

    edge_terms = []
    for start, end in tree.edges:
        start_d = _scaled_denominator(start, birth, death, sampling)
        end_d = _scaled_denominator(end, birth, death, sampling)
        edge_terms.append(
            -(birth - death) * (start - end)
            + 2.0 * (torch.log(end_d) - torch.log(start_d))
        )

    value = (
        tree.births * torch.log(birth)
        + tree.leaves * torch.log(sampling)
        + torch.stack(edge_terms).sum()
    )
    if normalize_root:
        value = value + torch.log(
            birth
            * _scaled_denominator(tree.root_time, birth, death, sampling)
        )
    return value


def _initial_raw(
    birth: float,
    death: float,
    sampling: float,
    max_birth: float,
    max_death: float,
) -> tuple[float, float, float]:
    death_fraction = death / max_death
    birth_fraction = (birth - death) / (max_birth - death)
    return (
        _logit(death_fraction),
        _logit(birth_fraction),
        _logit(sampling),
    )


def fit(
    tree: TreeData,
    max_birth: float = 1000.0,
    max_death: float = 500.0,
    max_iter: int = 2000,
) -> dict[str, object]:
    if max_birth <= max_death or max_death <= 0.0:
        raise ValueError("Require max_birth > max_death > 0")

    initializations = [
        (5.0, 1.0, 0.5),
        (10.0, 0.1, 0.05),
        (50.0, 0.1, 1e-6),
        (100.0, 1.0, 1e-10),
    ]
    best: tuple[float, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    restart_results = []

    for initial_birth, initial_death, initial_sampling in initializations:
        raw_death, raw_birth, raw_sampling = _initial_raw(
            initial_birth,
            initial_death,
            initial_sampling,
            max_birth,
            max_death,
        )
        parameters = [
            torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
            for value in (raw_death, raw_birth, raw_sampling)
        ]
        death_raw, birth_raw, sampling_raw = parameters
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def rates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            death = max_death * torch.sigmoid(death_raw)
            birth = death + (max_birth - death) * torch.sigmoid(birth_raw)
            sampling = torch.sigmoid(sampling_raw)
            return birth, death, sampling

        def objective() -> torch.Tensor:
            birth, death, sampling = rates()
            return calculate_log_likelihood(tree, birth, death, sampling)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            value = objective()
            loss = -value
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite likelihood during optimization")
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            value = objective()
            birth, death, sampling = rates()
            candidate = (
                float(value),
                birth.detach().clone(),
                death.detach().clone(),
                sampling.detach().clone(),
            )
        restart_results.append(
            {
                "initial_birth": initial_birth,
                "initial_death": initial_death,
                "initial_sampling": initial_sampling,
                "log_likelihood": candidate[0],
                "birth_rate": float(candidate[1]),
                "death_rate": float(candidate[2]),
                "sampling_probability": float(candidate[3]),
            }
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    best_ll, birth, death, sampling = best
    root_u = 1.0 - 1.0 / (
        birth
        * _scaled_denominator(tree.root_time, birth, death, sampling)
    )
    return {
        "log_likelihood": best_ll,
        "birth_rate": float(birth),
        "death_rate": float(death),
        "sampling_probability": float(sampling),
        "root_U": float(root_u),
        "root_normalization": float(
            torch.log(
                birth
                * _scaled_denominator(tree.root_time, birth, death, sampling)
            )
        ),
        "max_birth_bound": max_birth,
        "max_death_bound": max_death,
        "restart_results": restart_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit one birth-death-sampling process to the whole SGOOTR tree."
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
        "--output",
        type=Path,
        default=Path("data/sgootr/inference/inference_unpooled_one_state.json"),
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    tree = load_tree(args.nodes, args.edges)
    result: dict[str, object] = {
        "method": "whole tree treated as one state; no transitions",
        "root_normalized": True,
        "root_node": tree.root,
        "root_time": tree.root_time,
        "num_nodes": tree.num_nodes,
        "num_edges": len(tree.edges),
        "birth_events": tree.births,
        "sampled_leaves": tree.leaves,
        "zero_length_edges": tree.zero_length_edges,
        "multifurcating_nodes": tree.multifurcating_nodes,
        "polytomy_assumption": (
            "Each node with k children contributes k-1 binary birth events."
        ),
        **fit(tree, max_iter=args.max_iter),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    summary_path = args.output.with_suffix(".out")
    summary_path.write_text(
        "\n".join(
            [
                f"LL={result['log_likelihood']:.15f}",
                f"birth: {result['birth_rate']:.15g}",
                f"death: {result['death_rate']:.15g}",
                f"sampling: {result['sampling_probability']:.15g}",
                f"root: {tree.root} at time {tree.root_time:.15g}",
                f"birth_events: {tree.births}",
                f"sampled_leaves: {tree.leaves}",
            ]
        )
        + "\n"
    )
    print(summary_path.read_text(), end="")
    print(f"JSON results: {args.output}")


if __name__ == "__main__":
    main()
