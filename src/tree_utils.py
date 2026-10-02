# tree_utils.py

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch

from src.dtos import (
    Node,
    Edge,
    PreparedTree,
)


# ============================================================
# POSTORDER TRAVERSAL
# ============================================================


def build_postorder(
    root: int,
    children: list[list[int]],
) -> np.ndarray:
    """
    Build a postorder traversal of the tree.

    C++ equivalent:

        buildPostorder(
            tree.root,
            tree.nodes,
            tree.postorder
        );

    Returns
    -------
    np.ndarray
        Shape (N,), dtype int64.

    Notes
    -----
    An iterative implementation is used rather than recursive
    DFS to avoid Python recursion-depth limitations.
    """

    postorder = []

    # (node, visited)
    stack = [(root, False)]

    while stack:

        v, visited = stack.pop()

        if visited:

            postorder.append(v)

        else:

            # Add node back after all children.
            stack.append((v, True))

            # Reverse so that children are processed in their
            # original order.
            for child in reversed(children[v]):
                stack.append((child, False))

    return np.asarray(
        postorder,
        dtype=np.int64,
    )


def build_levels(
    root: int,
    children: list[list[int]],
) -> list[list[int]]:
    levels = []
    current = [root]

    while current:
        levels.append(current)
        current = [child for node_id in current for child in children[node_id]]

    return list(reversed(levels))


# ============================================================
# PREPARE TREE
# ============================================================
def prepare_tree(
    input_nodes: Sequence[Node],
    input_edges: Sequence[Edge],
    device: torch.device | str = "cpu",
) -> PreparedTree:
    """
    Convert raw Node/Edge objects into a PreparedTree.

    All persistent tree data remain NumPy arrays.
    Only `time_t` is converted to a PyTorch tensor.

    The input node IDs may be arbitrary. They are converted
    to contiguous internal indices:

        original ID -> [0, 1, ..., N-1]
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(input_nodes) == 0:
        raise ValueError("No input nodes were provided.")

    if len(input_edges) == 0:
        raise ValueError("No input edges were provided.")

    n = len(input_nodes)
    m = len(input_edges)

    device = torch.device(device)

    # ========================================================
    # ORIGINAL ID -> CONTIGUOUS INDEX
    # ========================================================

    id_to_idx = {node.id: i for i, node in enumerate(input_nodes)}

    if len(id_to_idx) != n:
        raise ValueError("Duplicate node IDs detected.")

    # ========================================================
    # NODE ARRAYS
    # ========================================================

    original_id = np.empty(
        n,
        dtype=np.int64,
    )

    time = np.empty(
        n,
        dtype=np.float64,
    )

    state = np.empty(
        n,
        dtype=np.int64,
    )

    parent = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    edge_idx_to_parent = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for i, node in enumerate(input_nodes):

        original_id[i] = node.id
        time[i] = node.T
        state[i] = node.state

    # ========================================================
    # EDGE ARRAYS
    # ========================================================

    edge_parent = np.empty(
        m,
        dtype=np.int64,
    )

    edge_child = np.empty(
        m,
        dtype=np.int64,
    )

    # Used only while constructing the tree.
    children = [[] for _ in range(n)]

    indegree = np.zeros(
        n,
        dtype=np.int64,
    )

    # ========================================================
    # PROCESS EDGES
    # ========================================================

    for edge_idx, edge in enumerate(input_edges):

        if edge.u not in id_to_idx:
            raise ValueError(f"Unknown node ID in edge.u: {edge.u}")

        if edge.v not in id_to_idx:
            raise ValueError(f"Unknown node ID in edge.v: {edge.v}")

        u = id_to_idx[edge.u]
        v = id_to_idx[edge.v]

        # ----------------------------------------------------
        # Store edge
        # ----------------------------------------------------

        edge_parent[edge_idx] = u
        edge_child[edge_idx] = v

        # ----------------------------------------------------
        # Store topology
        # ----------------------------------------------------

        children[u].append(v)

        if parent[v] != -1:
            raise ValueError(f"Node {edge.v} has multiple parents.")

        parent[v] = u
        edge_idx_to_parent[v] = edge_idx

        indegree[v] += 1

        if indegree[v] > 1:
            raise ValueError(f"Node {edge.v} has indegree > 1.")

    # ========================================================
    # ROOT
    # ========================================================

    roots = np.flatnonzero(indegree == 0)

    if len(roots) == 0:
        raise ValueError("No root found.")

    if len(roots) > 1:
        raise ValueError(f"Multiple roots detected: " f"{roots.tolist()}")

    root = int(roots[0])

    # ========================================================
    # POSTORDER
    # ========================================================

    postorder = build_postorder(
        root=root,
        children=children,
    )

    postorder = [int(node_id) for node_id in postorder]

    levels = build_levels(
        root=root,
        children=children,
    )

    level_batches = []

    for level in levels[:-1]:
        leaf_nodes = [node_id for node_id in level if not children[node_id]]
        internal_nodes = [node_id for node_id in level if children[node_id]]

        batch = {}

        if leaf_nodes:
            batch["leaf_edge_ids"] = torch.tensor(
                [int(edge_idx_to_parent[node_id]) for node_id in leaf_nodes],
                dtype=torch.long,
                device=device,
            )
            batch["leaf_states"] = torch.tensor(
                [int(state[node_id]) for node_id in leaf_nodes],
                dtype=torch.long,
                device=device,
            )
            batch["leaf_parent_times"] = torch.tensor(
                [float(time[parent[node_id]]) for node_id in leaf_nodes],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            batch["leaf_child_times"] = torch.tensor(
                [float(time[node_id]) for node_id in leaf_nodes],
                dtype=torch.get_default_dtype(),
                device=device,
            )

        if internal_nodes:
            batch["internal_edge_ids"] = torch.tensor(
                [int(edge_idx_to_parent[node_id]) for node_id in internal_nodes],
                dtype=torch.long,
                device=device,
            )
            batch["internal_child_edge_ids"] = torch.tensor(
                [
                    [
                        int(edge_idx_to_parent[children[node_id][0]]),
                        int(edge_idx_to_parent[children[node_id][1]]),
                    ]
                    for node_id in internal_nodes
                ],
                dtype=torch.long,
                device=device,
            )
            batch["internal_parent_times"] = torch.tensor(
                [float(time[parent[node_id]]) for node_id in internal_nodes],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            batch["internal_child_times"] = torch.tensor(
                [float(time[node_id]) for node_id in internal_nodes],
                dtype=torch.get_default_dtype(),
                device=device,
            )

        level_batches.append(batch)

    # --------------------------------------------------------
    # Verify that every node was reached.
    # --------------------------------------------------------

    if len(postorder) != n:
        raise ValueError(
            "Tree is disconnected or contains a cycle. "
            f"Expected {n} nodes in postorder, got "
            f"{len(postorder)}."
        )

    # ========================================================
    # TORCH TIME
    # ========================================================
    #
    # This is the only tree field converted to Torch. Static
    # topology remains in Python lists for graph capture.
    # ========================================================

    time_t = torch.as_tensor(
        time,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    # ========================================================
    # PREPARED TREE
    # ========================================================

    return PreparedTree(
        root=root,
        # Python static data
        original_id=original_id.tolist(),
        time=time.tolist(),
        state=state.tolist(),
        parent=parent.tolist(),
        children=children,
        edge_idx_to_parent=edge_idx_to_parent.tolist(),
        edge_parent=edge_parent.tolist(),
        edge_child=edge_child.tolist(),
        postorder=postorder,
        levels=levels,
        level_batches=level_batches,
        # Torch
        time_t=time_t,
    )
