from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TreeData:
    root_id: str
    children: dict[str, list[tuple[str, float]]]
    tips: tuple[str, ...]
    traversal_order: tuple[str, ...]
    node_ages: dict[str, float]


def _read_nonnegative_float(value: str, description: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{description} must be a number, got {value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return result


def read_tree(nodes_path: Path, edges_path: Path) -> TreeData:
    node_ids: list[str] = []
    declared_tips: set[str] = set()
    node_ages: dict[str, float] = {}

    with nodes_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"node_id", "T", "is_leaf"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{nodes_path} must contain columns: "
                f"{', '.join(sorted(required_columns))}"
            )
        for row in reader:
            node_id = (row["node_id"] or "").strip()
            if not node_id:
                raise ValueError(f"{nodes_path} contains an empty node_id")
            if node_id in node_ages:
                raise ValueError(f"{nodes_path} contains duplicate node_id {node_id!r}")
            node_ids.append(node_id)
            node_ages[node_id] = _read_nonnegative_float(
                row["T"] or "", f"age for node {node_id!r}"
            )
            leaf_value = (row["is_leaf"] or "").strip().lower()
            if leaf_value not in {"0", "1", "false", "true"}:
                raise ValueError(
                    f"is_leaf for node {node_id!r} must be 0 or 1, "
                    f"got {leaf_value!r}"
                )
            if leaf_value in {"1", "true"}:
                declared_tips.add(node_id)

    if not node_ids:
        raise ValueError(f"{nodes_path} contains no nodes")

    children = {node_id: [] for node_id in node_ids}
    parent_by_child: dict[str, str] = {}
    seen_edges: set[tuple[str, str]] = set()
    with edges_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"parent", "child", "branch_length"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{edges_path} must contain columns: "
                f"{', '.join(sorted(required_columns))}"
            )
        for row in reader:
            parent = (row["parent"] or "").strip()
            child = (row["child"] or "").strip()
            if parent not in children or child not in children:
                raise ValueError(
                    f"Edge {parent!r}->{child!r} references a node absent from "
                    f"{nodes_path}"
                )
            if parent == child or child in parent_by_child:
                raise ValueError(f"Invalid or duplicate parent for node {child!r}")
            edge = (parent, child)
            if edge in seen_edges:
                raise ValueError(f"Duplicate edge {parent!r}->{child!r}")
            seen_edges.add(edge)
            branch_length = _read_nonnegative_float(
                row["branch_length"] or "",
                f"branch length for edge {parent!r}->{child!r}",
            )
            parent_by_child[child] = parent
            children[parent].append((child, branch_length))

    roots = [node_id for node_id in node_ids if node_id not in parent_by_child]
    if len(roots) != 1:
        raise ValueError(f"Expected one rooted tree, found {len(roots)} roots")
    root_id = roots[0]

    traversal_order: list[str] = []
    stack = [root_id]
    visited: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            raise ValueError("The edge list contains a cycle")
        visited.add(node_id)
        traversal_order.append(node_id)
        stack.extend(child for child, _ in reversed(children[node_id]))
    if len(traversal_order) != len(node_ids):
        raise ValueError("The edge list is disconnected or contains a cycle")

    actual_tips = {node_id for node_id in node_ids if not children[node_id]}
    if actual_tips != declared_tips:
        raise ValueError(
            "nodes.csv is_leaf values do not match terminal nodes in edges.csv"
        )
    if not actual_tips:
        raise ValueError("The tree has no tips")

    return TreeData(
        root_id=root_id,
        children=children,
        tips=tuple(node_id for node_id in node_ids if node_id in actual_tips),
        traversal_order=tuple(traversal_order),
        node_ages=node_ages,
    )


def _sample_descendant_states(
    parent_states: np.ndarray,
    branch_length: float,
    mutation_rate: float,
    num_edit_states: int,
    silencing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    child_states = parent_states.copy()
    if branch_length == 0 or mutation_rate == 0:
        return child_states

    scaled_length = branch_length * mutation_rate
    probability_not_silenced = math.exp(-scaled_length * silencing_rate)
    probability_silenced = -math.expm1(-scaled_length * silencing_rate)
    probability_edited = probability_not_silenced * -math.expm1(-scaled_length)
    draws = rng.random(parent_states.size)

    unedited = parent_states == 0
    child_states[unedited & (draws < probability_silenced)] = -1
    edit_events = unedited & (draws >= probability_silenced) & (
        draws < probability_silenced + probability_edited
    )
    if np.any(edit_events):
        child_states[edit_events] = rng.integers(
            1,
            num_edit_states + 1,
            size=int(np.count_nonzero(edit_events)),
            dtype=np.int64,
        )

    edited = parent_states > 0
    child_states[edited & (draws >= probability_not_silenced)] = -1
    return child_states


def simulate_barcode_matrix(
    nodes_path: Path,
    edges_path: Path,
    num_sites: int,
    mutation_rate: float,
    num_edit_states: int,
    silencing_rate: float = 0.0,
    origin_time: float | None = None,
    seed: int | None = None,
) -> tuple[tuple[str, ...], list[list[int]]]:
    if num_sites < 1:
        raise ValueError("num_sites must be at least 1")
    if num_edit_states < 1:
        raise ValueError("num_edit_states must be at least 1")
    mutation_rate = _read_nonnegative_float(str(mutation_rate), "mutation_rate")
    silencing_rate = _read_nonnegative_float(
        str(silencing_rate), "silencing_rate"
    )
    if origin_time is not None:
        origin_time = _read_nonnegative_float(str(origin_time), "origin_time")

    tree = read_tree(nodes_path, edges_path)
    rng = np.random.default_rng(seed)
    states_by_node: dict[str, np.ndarray] = {
        tree.root_id: np.zeros(num_sites, dtype=np.int64)
    }

    if origin_time is not None:
        stem_length = origin_time - tree.node_ages[tree.root_id]
        if stem_length < 0:
            raise ValueError(
                "origin_time must be at least the root age in nodes.csv"
            )
        if stem_length > 0:
            states_by_node[tree.root_id] = _sample_descendant_states(
                states_by_node[tree.root_id],
                stem_length,
                mutation_rate,
                num_edit_states,
                silencing_rate,
                rng,
            )

    for parent in tree.traversal_order:
        parent_states = states_by_node[parent]
        for child, branch_length in tree.children[parent]:
            states_by_node[child] = _sample_descendant_states(
                parent_states,
                branch_length,
                mutation_rate,
                num_edit_states,
                silencing_rate,
                rng,
            )

    matrix = np.stack([states_by_node[tip] for tip in tree.tips]).tolist()
    return tree.tips, matrix


def write_barcode_matrix(
    output_path: Path,
    tip_ids: tuple[str, ...],
    matrix: list[list[int]],
) -> None:
    if len(tip_ids) != len(matrix):
        raise ValueError("tip_ids and matrix must contain the same number of rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["", *(f"barcode_{index + 1}" for index in range(len(matrix[0])))])
        for tip_id, states in zip(tip_ids, matrix):
            writer.writerow([tip_id, *states])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate BEAM-format CRISPR barcode data down a rooted "
            "nodes.csv/edges.csv tree."
        )
    )
    parser.add_argument("--nodes", type=Path, required=True, help="Tree nodes.csv")
    parser.add_argument("--edges", type=Path, required=True, help="Tree edges.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output barcode TSV")
    parser.add_argument(
        "--num-sites",
        type=int,
        required=True,
        help="Number of independent barcode sites to simulate.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        required=True,
        help="CRISPR clock rate per branch-length time unit.",
    )
    parser.add_argument(
        "--num-edit-states",
        type=int,
        default=4,
        help="Number of equiprobable edit outcomes per site (default: 4).",
    )
    parser.add_argument(
        "--silencing-rate",
        type=float,
        default=0.0,
        help="BEAM silencing rate relative to the CRISPR clock (default: 0).",
    )
    parser.add_argument(
        "--origin-time",
        type=float,
        help=(
            "Optional BEAM origin age in the same units as nodes.csv T; "
            "a longer origin-to-root stem is simulated when supplied."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1).",
    )
    args = parser.parse_args()

    tip_ids, matrix = simulate_barcode_matrix(
        nodes_path=args.nodes,
        edges_path=args.edges,
        num_sites=args.num_sites,
        mutation_rate=args.mutation_rate,
        num_edit_states=args.num_edit_states,
        silencing_rate=args.silencing_rate,
        origin_time=args.origin_time,
        seed=args.seed,
    )
    write_barcode_matrix(args.output, tip_ids, matrix)
    print(
        f"Wrote {len(tip_ids)} tips x {args.num_sites} barcode sites "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
