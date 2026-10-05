# tree_utils.py

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

import numpy as np
import torch
import networkx as nx

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


def prepare_nx_tree(
    input_nodes: Sequence[Node],
    input_edges: Sequence[Edge],
) -> nx.DiGraph:
    """Build a NetworkX directed tree from raw node/edge records."""

    if len(input_nodes) == 0:
        raise ValueError("No input nodes were provided.")

    if len(input_edges) == 0:
        raise ValueError("No input edges were provided.")

    graph = nx.DiGraph()
    node_ids = {node.id for node in input_nodes}

    for node in input_nodes:
        graph.add_node(
            node.id,
            T=float(node.T),
            is_leaf=bool(node.is_leaf),
            state=int(node.state),
        )

    indegree = {node_id: 0 for node_id in node_ids}

    for edge in input_edges:
        if edge.u not in node_ids:
            raise ValueError(f"Unknown node ID in edge.u: {edge.u}")

        if edge.v not in node_ids:
            raise ValueError(f"Unknown node ID in edge.v: {edge.v}")

        if edge.u == edge.v:
            raise ValueError(f"Self-loop edge detected: {edge.u}")

        graph.add_edge(edge.u, edge.v, length=float(edge.length))
        indegree[edge.v] += 1

    roots = [node_id for node_id, degree in indegree.items() if degree == 0]

    if len(roots) == 0:
        raise ValueError("No root found.")

    if len(roots) > 1:
        raise ValueError(f"Multiple roots detected: {roots}")

    graph.graph["root"] = roots[0]

    return graph


def cut_tree(
    tree: nx.DiGraph,
    cut_times: Sequence[float],
) -> tuple[list[list[nx.DiGraph]], nx.DiGraph]:
    """Split a rooted tree by time cuts into nested components.

    Each level stores the nodes between the current cut and the next lower cut,
    so deeper levels do not inherit nodes from shallower levels.

    Edges that cross a cut are split at the cut point: a new node carrying the
    node attribute ``cut=True`` is inserted at the cut time. The cut node is
    added as a leaf of the subtree above the cut and, under the same name, as
    the root of the descendant subtree below the cut.
    """

    if not cut_times:
        raise ValueError("No cut times were provided.")

    if "root" not in tree.graph:
        raise ValueError("Tree does not have tree.graph['root'].")

    root = tree.graph["root"]
    if root not in tree:
        raise ValueError(f"Root {root} is not present in the tree.")

    cuts_desc = sorted({float(t) for t in cut_times}, reverse=True)
    node_times = {node: float(data["T"]) for node, data in tree.nodes(data=True)}

    for u, v in tree.edges():
        Tu = node_times[u]
        Tv = node_times[v]
        if Tu < Tv:
            raise ValueError(
                f"Invalid temporal ordering on edge {u} -> {v}: "
                f"T[{u}]={Tu} < T[{v}]={Tv}"
            )

    def descendants(start: int) -> set[int]:
        seen: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(tree.successors(node))
        return seen

    def cut_node_name(u: int, v: int, cut_time: float) -> str:
        return f"cut_{u}_{v}_{cut_time}"

    def add_cut_node(
        subgraph: nx.DiGraph,
        u: int,
        v: int,
        cut_time: float,
        is_leaf: bool,
    ) -> str:
        name = cut_node_name(u, v, cut_time)
        subgraph.add_node(
            name,
            T=float(cut_time),
            is_leaf=is_leaf,
            state=tree.nodes[v].get("state"),
            cut=True,
        )
        return name

    def crossing_edges(cut_time: float) -> list[tuple[int, int]]:
        return [
            (u, v)
            for u, v in tree.edges()
            if node_times[u] > cut_time >= node_times[v]
        ]

    level_trees: list[list[nx.DiGraph]] = [[] for _ in range(len(cuts_desc) + 1)]
    hierarchy = nx.DiGraph()

    # Maps a cut node name to the hierarchy key of the tree in which the
    # cut node appears as a leaf at the lower boundary of that tree.
    cut_leaf_key = {}

    # --------------------------------------------------------
    # Level 0: nodes above the highest cut, plus cut nodes at
    # the cut points of every edge crossing the cut.
    # --------------------------------------------------------

    top_cut = cuts_desc[0]
    top_nodes = {node for node, T in node_times.items() if T > top_cut}
    if not top_nodes:
        raise ValueError(
            f"No nodes remain above the first cut time {top_cut}; "
            "the root-side subtree is empty."
        )

    top_tree = tree.subgraph(top_nodes).copy()

    root_key = (0, root)

    for u, v in crossing_edges(top_cut):
        name = add_cut_node(top_tree, u, v, top_cut, is_leaf=True)
        top_tree.add_edge(u, name, length=node_times[u] - top_cut)
        cut_leaf_key[name] = root_key

    top_tree.graph["root"] = root
    top_tree.graph["cut_time"] = top_cut
    top_tree.graph["level"] = 0
    level_trees[0].append(top_tree)

    hierarchy.add_node(
        root_key,
        level=0,
        root=root,
        cut_time=top_cut,
        tree=top_tree,
    )

    # --------------------------------------------------------
    # Levels 1..len(cuts): subtrees below each cut, rooted at
    # the cut node of the edge crossing the cut.
    # --------------------------------------------------------

    for level, cut_time in enumerate(cuts_desc):
        next_cut = cuts_desc[level + 1] if level + 1 < len(cuts_desc) else None

        for u, v in crossing_edges(cut_time):
            allowed = {
                node
                for node in descendants(v)
                if next_cut is None or node_times[node] > next_cut
            }

            subtree = tree.subgraph(allowed).copy()

            # The cut node on edge (u, v) is the root of the descendant
            # subtree, shared under the same name with the parent subtree.
            root_name = add_cut_node(subtree, u, v, cut_time, is_leaf=False)
            if v in allowed:
                subtree.add_edge(
                    root_name,
                    v,
                    length=cut_time - node_times[v],
                )

            child_key = (level + 1, cut_node_name(u, v, cut_time))

            # Cut nodes at the lower boundary of this slab.
            if next_cut is not None:
                for x, y in crossing_edges(next_cut):
                    if x in allowed:
                        name = add_cut_node(
                            subtree, x, y, next_cut, is_leaf=True
                        )
                        subtree.add_edge(
                            x, name, length=node_times[x] - next_cut
                        )
                        cut_leaf_key[name] = child_key
                    elif x == u and y == v:
                        # The root segment itself crosses the lower cut.
                        name = add_cut_node(
                            subtree, x, y, next_cut, is_leaf=True
                        )
                        subtree.add_edge(
                            root_name, name, length=cut_time - next_cut
                        )
                        cut_leaf_key[name] = child_key

            subtree.graph["root"] = root_name
            subtree.graph["cut_time"] = cut_time
            subtree.graph["level"] = level + 1
            level_trees[level + 1].append(subtree)

            hierarchy.add_node(
                child_key,
                level=level + 1,
                root=root_name,
                cut_time=cut_time,
                tree=subtree,
            )
            # The parent is the tree in which this subtree's root cut node
            # appears as a leaf at the lower boundary.
            hierarchy.add_edge(cut_leaf_key[root_name], child_key)

    for trees in level_trees:
        trees.sort(key=lambda g: str(g.graph["root"]))

    return level_trees, hierarchy


def get_datastructure_for_likelihood_calculation(
    levels: list[list[nx.DiGraph]],
    hierarchy: nx.DiGraph,
    cut_times: list[float],
    num_states: int,
):
    level_ds = []

    for level in levels:
        ds = {}
        num_births_of_level_trees = []
        num_sampling_of_level_trees = []
        sampling_state_counts_of_level_trees = []
        start_times_of_level_trees = []
        end_times_of_level_trees = []
        
        for tree in level:
            num_births = 0
            sampling_state_counts = np.zeros(num_states, dtype=np.int64)

            for n, d in tree.nodes(data=True):
                if d.get("cut") is None:
                    if d["is_leaf"]:
                        state = int(d["state"])
                        if not 0 <= state < num_states:
                            raise ValueError(
                                f"Leaf state {state} is outside [0, {num_states})."
                            )
                        sampling_state_counts[state] += 1
                    elif n != tree.graph["root"]:
                        num_births += 1

            num_sampling = int(sampling_state_counts.sum())
            num_births_of_level_trees.append(num_births)
            num_sampling_of_level_trees.append(num_sampling)
            sampling_state_counts_of_level_trees.append(sampling_state_counts)

            start_times = []
            end_times = []

            for u, v in tree.edges():
                start_time = tree.nodes[u]["T"]
                end_time = tree.nodes[v]["T"]
                start_times.append(start_time)
                end_times.append(end_time)

            start_times_of_level_trees.append(start_times)
            end_times_of_level_trees.append(end_times)

        edge_matrix = np.zeros((len(level), sum(len(st) for st in start_times_of_level_trees)))
        cum_start_times = 0
        for tree_idx, tree in enumerate(level):
            edge_matrix[tree_idx, cum_start_times:cum_start_times + len(start_times_of_level_trees[tree_idx])] = 1
            cum_start_times += len(start_times_of_level_trees[tree_idx])
            
        
        ds["num_births"] = torch.as_tensor(num_births_of_level_trees)
        ds["num_sampling"] = torch.as_tensor(num_sampling_of_level_trees)
        ds["num_sampling_by_state"] = torch.as_tensor(
            np.asarray(sampling_state_counts_of_level_trees, dtype=np.int64)
        )
        ds["start_times"] = torch.as_tensor([st for st_list in start_times_of_level_trees for st in st_list])
        ds["end_times"] = torch.as_tensor([et for et_list in end_times_of_level_trees for et in et_list])
        ds["edge_matrix"] = torch.as_tensor(edge_matrix)

        level_ds.append(ds)

    for i in range(len(levels) - 1):
        
        current_level_tree_indices = {}
        for idx, tree in enumerate(levels[i]):
            current_level_tree_indices[tree.graph["root"]] = idx
        next_level_tree_indices = {}
        for idx, tree in enumerate(levels[i + 1]):
            next_level_tree_indices[tree.graph["root"]] = idx

        transition_matrix = np.zeros((len(current_level_tree_indices), len(next_level_tree_indices)))

        for u, v in hierarchy.edges():
            if u[0] == i and v[0] == i + 1 and \
                u[1] in current_level_tree_indices and v[1] in next_level_tree_indices:
                
                transition_matrix[current_level_tree_indices[u[1]], next_level_tree_indices[v[1]]] = 1

        level_ds[i]["transition_matrix"] = torch.as_tensor(transition_matrix)

    print(level_ds)
    
    return level_ds
