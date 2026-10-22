from __future__ import annotations

import copy
import math

from Bio.Phylo.BaseTree import Tree

from src.process_data.sgootr.tree import TreeRepresentation, build_tree_representation


def anchor_birth_events_to_transitions(
    source: TreeRepresentation,
    transition_times: list[float],
    minimum_terminal_branch_length: float,
) -> tuple[Tree, float]:
    if not transition_times:
        raise ValueError("At least one transition time is required for anchoring.")
    if (
        not math.isfinite(minimum_terminal_branch_length)
        or minimum_terminal_branch_length <= 0.0
    ):
        raise ValueError(
            "minimum_terminal_branch_length must be finite and positive."
        )

    latest_transition = max(transition_times)
    if latest_transition < 0.0 or latest_transition > source.root_age:
        raise ValueError(
            f"Latest transition time {latest_transition:.10g} is outside "
            f"the tree age [0, {source.root_age:.10g}]."
        )

    post_transition_birth_nodes = [
        node
        for node, data in source.graph.nodes(data=True)
        if node != source.root
        and not data["terminal"]
        and source.depths[node] > latest_transition + 1e-12
    ]
    scale_limits = []
    for leaf, data in source.graph.nodes(data=True):
        if not data["terminal"]:
            continue
        parents = list(source.graph.predecessors(leaf))
        if len(parents) != 1:
            raise ValueError(f"Terminal node {leaf} must have one parent.")
        parent = parents[0]
        parent_depth = source.depths[parent]
        leaf_depth = source.depths[leaf]
        if parent_depth > latest_transition + 1e-12:
            scale_limits.append(
                (
                    (
                        leaf_depth
                        - minimum_terminal_branch_length
                        - latest_transition
                    )
                    / (parent_depth - latest_transition),
                    parent,
                    leaf,
                )
            )

    if post_transition_birth_nodes and not scale_limits:
        raise ValueError(
            "Cannot preserve the terminal-branch floor after the latest transition."
        )
    if scale_limits:
        scale_factor, limiting_parent, limiting_leaf = min(
            scale_limits, key=lambda item: item[0]
        )
        if not math.isfinite(scale_factor) or scale_factor < 1.0 - 1e-10:
            raise ValueError(
                "No expansion factor of at least one preserves fixed tip dates "
                f"and the {minimum_terminal_branch_length:g} terminal-branch floor."
            )
        scale_factor = max(1.0, scale_factor)
    else:
        scale_factor = 1.0
        limiting_parent = limiting_leaf = None

    adjusted_depths = dict(source.depths)
    for node in post_transition_birth_nodes:
        adjusted_depths[node] = latest_transition + scale_factor * (
            source.depths[node] - latest_transition
        )

    adjusted_tree = copy.deepcopy(source.tree)
    adjusted_clades = tuple(adjusted_tree.find_clades(order="preorder"))
    terminal_lengths = []
    for parent in source.clades:
        parent_node = source.node_id_by_clade_id[id(parent)]
        for child in parent.clades:
            child_node = source.node_id_by_clade_id[id(child)]
            length = adjusted_depths[child_node] - adjusted_depths[parent_node]
            if length < -1e-10:
                raise ValueError(
                    f"Birth-event scaling reverses edge {parent_node}->{child_node}."
                )
            adjusted_clades[child_node].branch_length = max(0.0, length)
            if child.is_terminal() and (
                source.depths[child_node] > latest_transition + 1e-12
            ):
                terminal_lengths.append(length)
    adjusted_tree.root.branch_length = None

    if terminal_lengths and min(terminal_lengths) < (
        minimum_terminal_branch_length - 1e-10
    ):
        raise ValueError(
            "The transition-anchored tree violates the post-transition "
            "terminal-branch floor."
        )

    adjusted_stage = build_tree_representation(adjusted_tree)
    if not math.isclose(
        adjusted_stage.root_age,
        source.root_age,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Transition anchoring changed the root-to-tip age.")
    if limiting_parent is not None and limiting_leaf is not None:
        print(
            f"Terminal branch constraint set by edge "
            f"{limiting_parent}->{limiting_leaf}."
        )
    return adjusted_tree, scale_factor
