from __future__ import annotations

import copy
import math

from Bio.Phylo.BaseTree import Tree

from src.process_data.sgootr.tree import build_tree_representation


def shift_birth_events(
    tree: Tree,
    minimum_branch_length: float,
    ultrametric_tolerance: float = 1e-5,
) -> Tree:
    if not math.isfinite(minimum_branch_length) or minimum_branch_length <= 0.0:
        raise ValueError("minimum_branch_length must be finite and positive")

    stage = build_tree_representation(tree)
    tip_depths = [
        stage.depths[stage.node_id_by_clade_id[id(clade)]]
        for clade in stage.clades
        if clade.is_terminal()
    ]
    if max(tip_depths) - min(tip_depths) > ultrametric_tolerance:
        raise ValueError("Birth-event shifting requires an ultrametric input tree.")

    original_times = {
        node: stage.root_age - depth for node, depth in stage.depths.items()
    }
    adjusted_times: dict[int, float] = {}
    for clade in reversed(stage.clades):
        node = stage.node_id_by_clade_id[id(clade)]
        if clade.is_terminal():
            adjusted_times[node] = 0.0
            continue

        child_nodes = [
            stage.node_id_by_clade_id[id(child)] for child in clade.clades
        ]
        latest_allowed_time = max(
            adjusted_times[child] + minimum_branch_length
            for child in child_nodes
        )
        original_time = original_times[node]
        if latest_allowed_time > original_time + 1e-10:
            raise ValueError(
                f"Cannot enforce a {minimum_branch_length:g} minimum branch "
                f"length while preserving root age at node {node}: it would "
                f"need time {latest_allowed_time:.10g}, later than "
                f"{original_time:.10g}."
            )
        adjusted_times[node] = (
            original_time if node == stage.root else latest_allowed_time
        )

    adjusted_tree = copy.deepcopy(tree)
    adjusted_clades = tuple(adjusted_tree.find_clades(order="preorder"))
    for parent in stage.clades:
        parent_node = stage.node_id_by_clade_id[id(parent)]
        for child in parent.clades:
            child_node = stage.node_id_by_clade_id[id(child)]
            branch_length = adjusted_times[parent_node] - adjusted_times[child_node]
            if branch_length < minimum_branch_length - 1e-10:
                raise ValueError(
                    f"Adjusted edge {parent_node}->{child_node} is too short: "
                    f"{branch_length:.15g}."
                )
            adjusted_clades[child_node].branch_length = branch_length
    adjusted_tree.root.branch_length = None
    return adjusted_tree
