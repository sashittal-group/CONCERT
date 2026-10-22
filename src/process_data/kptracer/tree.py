from __future__ import annotations

import csv
import math
from pathlib import Path

from Bio import Phylo
import networkx as nx
import pandas as pd

from src.dataprocessor import export_tree
from src.likelihood import add_node_times


STATE_CODES = {
    "T1": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "S1": 4,
}


def build_tree_graph(
    newick_path: Path,
    metadata_path: Path,
    *,
    state_column: str = "SubTumor",
) -> nx.DiGraph:
    tree = Phylo.read(str(newick_path), "newick")
    clades = list(tree.find_clades(order="level"))
    node_ids = {clade: index + 1 for index, clade in enumerate(clades)}

    graph = nx.DiGraph()
    for clade, node_id in node_ids.items():
        graph.add_node(node_id, name=clade.name)
    for parent in clades:
        for child in parent.clades:
            branch_length = child.branch_length
            if branch_length is None or not math.isfinite(branch_length):
                raise ValueError(
                    f"Tree edge to {child.name!r} has no finite branch length"
                )
            if branch_length < 0.0:
                raise ValueError(
                    f"Tree edge to {child.name!r} has negative branch length"
                )
            graph.add_edge(
                node_ids[parent],
                node_ids[child],
                length=float(branch_length),
            )

    metadata = pd.read_csv(metadata_path, index_col=0)
    if state_column not in metadata.columns:
        raise ValueError(f"{metadata_path} is missing column {state_column!r}")
    metadata.index = metadata.index.map(str)
    if not metadata.index.is_unique:
        raise ValueError(f"Metadata index in {metadata_path} must be unique")

    for clade in clades:
        node_name = str(clade)
        if node_name not in metadata.index:
            continue
        state_value = metadata.at[node_name, state_column]
        if pd.isna(state_value):
            continue
        state_code = str(state_value)[8:10]
        if state_code not in STATE_CODES:
            raise ValueError(
                f"Unsupported {state_column} value {state_value!r} "
                f"for tree node {node_name!r}"
            )
        graph.nodes[node_ids[clade]]["state"] = STATE_CODES[state_code]
    root_id = node_ids[tree.root]
    root_id = node_ids[tree.root]
    graph.graph["root"] = root_id
    return graph


def write_initial_tree_exports(graph: nx.DiGraph, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "1461.tree"
    label_path = output_dir / "1461.labeling"

    with edge_path.open("w", newline="") as edge_file:
        writer = csv.writer(edge_file, delimiter="\t", lineterminator="\n")
        writer.writerows(graph.edges())

    with label_path.open("w", newline="") as label_file:
        writer = csv.writer(label_file, delimiter="\t", lineterminator="\n")
        writer.writerows(
            (node, data["state"])
            for node, data in graph.nodes(data=True)
            if "state" in data
        )

    return {"tree_edges": edge_path, "leaf_labels": label_path}


def normalize_tree(
    graph: nx.DiGraph,
    *,
    minimum_edge_length: float = 0.001,
    target_tip_depth: float = 1.0,
) -> nx.DiGraph:
    if not math.isfinite(minimum_edge_length) or minimum_edge_length <= 0.0:
        raise ValueError("minimum_edge_length must be a positive finite value")
    if not math.isfinite(target_tip_depth) or target_tip_depth <= 0.0:
        raise ValueError("target_tip_depth must be a positive finite value")

    root = graph.graph.get("root", 0)
    processed = graph.copy()
    add_node_times(processed, root=root)
    initial_depths = nx.single_source_dijkstra_path_length(
        processed,
        root,
        weight="length",
    )
    maximum_depth = max(initial_depths.values())
    if not math.isfinite(maximum_depth) or maximum_depth <= 0.0:
        raise ValueError("Tree must have a positive finite root-to-tip depth")

    for parent, child in processed.edges():
        processed.edges[parent, child]["length"] /= maximum_depth

    _extend_short_terminal_edges(processed, root, target_tip_depth)
    add_node_times(processed, root=root)
    processed = _enforce_minimum_edge_length(
        processed,
        root,
        minimum_edge_length,
    )
    _extend_short_terminal_edges(processed, root, target_tip_depth)
    add_node_times(processed, root=root)

    for node in processed.nodes:
        processed.nodes[node]["is_leaf"] = processed.out_degree(node) == 0
    return processed


def _extend_short_terminal_edges(
    graph: nx.DiGraph,
    root: int,
    target_tip_depth: float,
) -> None:
    depths = nx.single_source_dijkstra_path_length(
        graph,
        root,
        weight="length",
    )
    for leaf in (node for node in graph if graph.out_degree(node) == 0):
        distance = depths[leaf]
        if distance >= target_tip_depth:
            continue
        parents = list(graph.predecessors(leaf))
        if len(parents) != 1:
            raise ValueError(f"Expected one parent for terminal node {leaf}")
        parent = parents[0]
        graph.edges[parent, leaf]["length"] += target_tip_depth - distance


def _enforce_minimum_edge_length(
    graph: nx.DiGraph,
    root: int,
    minimum_edge_length: float,
) -> nx.DiGraph:
    processed = graph.copy()
    for node in nx.dfs_postorder_nodes(processed, root):
        children = list(processed.successors(node))
        if not children:
            continue
        required_time = max(
            processed.nodes[child]["T"] + minimum_edge_length
            for child in children
        )
        if processed.nodes[node]["T"] < required_time:
            processed.nodes[node]["T"] = required_time

    for parent, child in processed.edges():
        processed.edges[parent, child]["length"] = (
            processed.nodes[parent]["T"] - processed.nodes[child]["T"]
        )
    return processed


def apply_location_states(
    graph: nx.DiGraph,
    location_labels_path: Path,
) -> dict[int, int]:
    labels = pd.read_csv(
        location_labels_path,
        sep="\t",
        header=None,
        names=["node_id", "state"],
    )
    if labels["node_id"].duplicated().any():
        raise ValueError(f"Duplicate node IDs in {location_labels_path}")

    state_by_node = {
        int(node_id): int(state)
        for node_id, state in labels.itertuples(index=False, name=None)
    }
    graph_nodes = set(graph.nodes)
    unknown_nodes = set(state_by_node).difference(graph_nodes)
    missing_nodes = graph_nodes.difference(state_by_node)
    if unknown_nodes == {0}:
        state_by_node.pop(0)
        unknown_nodes.clear()
    if unknown_nodes or missing_nodes:
        raise ValueError(
            f"Location labels do not match the tree: "
            f"{len(unknown_nodes)} unknown and {len(missing_nodes)} missing nodes"
        )

    for node, state in state_by_node.items():
        graph.nodes[node]["state"] = state
    return state_by_node


def export_processed_tree(
    graph: nx.DiGraph,
    nodes_path: Path,
    edges_path: Path,
) -> None:
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    export_tree(graph, nodes_path, edges_path)
