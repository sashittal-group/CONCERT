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


# ============================================================
# PREPARE TREE
# ============================================================

def prepare_tree(
    input_nodes: Sequence[Node],
    input_edges: Sequence[Edge],
    device: torch.device | str = "cpu",
) -> PreparedTree:
    """
    Convert raw Node/Edge objects into a GPU-friendly
    PreparedTree.

    Equivalent to the C++ prepareTree().

    The input node IDs may be arbitrary. They are converted
    to contiguous internal indices:

        original ID -> [0, 1, ..., N-1]

    The resulting PreparedTree contains tensors for all
    parameter-independent numerical data.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(input_nodes) == 0:
        raise ValueError(
            "No input nodes were provided."
        )

    if len(input_edges) == 0:
        raise ValueError(
            "No input edges were provided."
        )

    n = len(input_nodes)
    m = len(input_edges)

    device = torch.device(device)

    # ========================================================
    # ORIGINAL ID -> CONTIGUOUS INDEX
    # ========================================================

    id_to_idx = {
        node.id: i
        for i, node in enumerate(input_nodes)
    }

    if len(id_to_idx) != n:
        raise ValueError(
            "Duplicate node IDs detected."
        )

    # ========================================================
    # NODE ARRAYS
    # ========================================================

    original_id = np.empty(
        n,
        dtype=np.int64,
    )

    times = np.empty(
        n,
        dtype=np.float64,
    )

    states = np.empty(
        n,
        dtype=np.int64,
    )

    # parent[i] = parent node index
    parent = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    # edge_idx_to_parent[i] =
    # index of edge connecting parent -> i
    edge_idx_to_parent = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for i, node in enumerate(input_nodes):

        original_id[i] = node.id
        times[i] = node.T
        states[i] = node.state

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

    # Python lists are used only during preprocessing.
    # They are converted into compact arrays afterward.
    children: list[list[int]] = [
        []
        for _ in range(n)
    ]

    # indegree is needed to identify the root.
    indegree = np.zeros(
        n,
        dtype=np.int64,
    )

    # ========================================================
    # PROCESS EDGES
    # ========================================================

    for edge_idx, edge in enumerate(input_edges):

        # ----------------------------------------------------
        # Parent
        # ----------------------------------------------------

        if edge.u not in id_to_idx:
            raise ValueError(
                f"Unknown node ID in edge.u: {edge.u}"
            )

        # ----------------------------------------------------
        # Child
        # ----------------------------------------------------

        if edge.v not in id_to_idx:
            raise ValueError(
                f"Unknown node ID in edge.v: {edge.v}"
            )

        u = id_to_idx[edge.u]
        v = id_to_idx[edge.v]

        # ----------------------------------------------------
        # Store edge
        # ----------------------------------------------------

        edge_parent[edge_idx] = u
        edge_child[edge_idx] = v

        # ----------------------------------------------------
        # Store tree structure
        # ----------------------------------------------------

        children[u].append(v)

        if parent[v] != -1:
            raise ValueError(
                f"Node {edge.v} has multiple parents."
            )

        parent[v] = u
        edge_idx_to_parent[v] = edge_idx

        indegree[v] += 1

        # A rooted tree should have at most one incoming
        # edge per node.
        if indegree[v] > 1:
            raise ValueError(
                f"Node {edge.v} has indegree > 1."
            )

    # ========================================================
    # ROOT
    # ========================================================

    roots = np.flatnonzero(
        indegree == 0
    )

    if len(roots) == 0:
        raise ValueError(
            "No root found."
        )

    if len(roots) > 1:
        raise ValueError(
            f"Multiple roots detected: "
            f"{roots.tolist()}"
        )

    root = int(roots[0])

    # ========================================================
    # POSTORDER
    # ========================================================

    postorder = build_postorder(
        root=root,
        children=children,
    )

    # --------------------------------------------------------
    # Verify that every node was reached.
    # This also catches disconnected/cyclic structures.
    # --------------------------------------------------------

    if len(postorder) != n:
        raise ValueError(
            "Tree is disconnected or contains a cycle. "
            f"Expected {n} nodes in postorder, got "
            f"{len(postorder)}."
        )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    # from_numpy() avoids an unnecessary Python-level copy.
    original_id_tensor = torch.from_numpy(
        original_id
    ).to(device=device)

    time_tensor = torch.from_numpy(
        times
    ).to(device=device)

    state_tensor = torch.from_numpy(
        states
    ).to(device=device)

    parent_tensor = torch.from_numpy(
        parent
    ).to(device=device)

    edge_idx_tensor = torch.from_numpy(
        edge_idx_to_parent
    ).to(device=device)

    edge_parent_tensor = torch.from_numpy(
        edge_parent
    ).to(device=device)

    edge_child_tensor = torch.from_numpy(
        edge_child
    ).to(device=device)

    postorder_tensor = torch.from_numpy(
        postorder
    ).to(device=device)

    # ========================================================
    # PREPARED TREE
    # ========================================================

    return PreparedTree(
        root=root,

        original_id=original_id_tensor,

        time=time_tensor,

        state=state_tensor,

        parent=parent_tensor,

        edge_idx_to_parent=edge_idx_tensor,

        edge_parent=edge_parent_tensor,

        edge_child=edge_child_tensor,

        postorder=postorder_tensor,
    )


# ============================================================
# OPTIONAL: MOVE PREPARED TREE
# ============================================================

def move_tree_to_device(
    tree: PreparedTree,
    device: torch.device | str,
) -> PreparedTree:
    """
    Move a PreparedTree between CPU, CUDA, and MPS.

    This is useful when trees are initially constructed on CPU
    and moved to GPU immediately before optimization.
    """

    device = torch.device(device)

    return PreparedTree(
        root=tree.root,

        original_id=tree.original_id.to(device),

        time=tree.time.to(device),

        state=tree.state.to(device),

        parent=tree.parent.to(device),

        edge_idx_to_parent=(
            tree.edge_idx_to_parent.to(device)
        ),

        edge_parent=tree.edge_parent.to(device),

        edge_child=tree.edge_child.to(device),

        postorder=tree.postorder.to(device),
    )