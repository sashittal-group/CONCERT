import csv

STATE_TO_ID = {
    "PGCLC": 0,
    "PCGLC": 0,
    "Endothelial": 1,
    "Endoderm": 2,
    "NMPs": 3,
    "Somite": 4,
    "NeuralTube": 5,
    "MP": 6,
    "NMSTD": 7,
    "NMSD": 8,
    "NMS": 9,
    "NS": 10,
}

def export_tree(T, node_file="nodes.csv", edge_file="edges.csv"):
    # --- write nodes ---
    with open(node_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "T", "is_leaf", "state_id"])

        for n, data in T.nodes(data=True):
            state = data.get("state")

            # hard mapping
            if state is None or state == "?":
                sid = -1
            else:
                # if state not in STATE_TO_ID:
                #     raise ValueError(f"Unknown state: {state}")
                # sid = STATE_TO_ID[state]
                sid = state

            writer.writerow([
                n,
                float(data["T"]),
                int(data.get("is_leaf", False)),
                sid
            ])

    # --- write edges ---
    with open(edge_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "branch_length"])

        for u, v, data in T.edges(data=True):
            writer.writerow([
                u,
                v,
                float(data.get("length", 0.0))
            ])



def export_diff_T(
    G,
    id_map,                     # dict: old_id -> new_id (partial)
    node_file="nodes.csv",
    edge_file="edges.csv",
    start_id=0                  # where to start incremental IDs
):
    """
    Export graph to CSV while remapping node IDs:
      - use id_map when present
      - otherwise assign incremental integers
    """

    # ---- Build complete mapping ----
    new_map = dict(id_map)  # copy so we don't mutate input
    next_id = start_id

    for node in G.nodes():
        if node not in new_map:
            # ensure no collision with existing mapped values
            while next_id in new_map.values():
                next_id += 1
            new_map[node] = next_id
            next_id += 1

    # ---- Collect node attributes ----
    node_attr_keys = set()
    for _, data in G.nodes(data=True):
        node_attr_keys.update(data.keys())
    node_attr_keys = sorted(node_attr_keys)

    # ---- Write nodes ----
    with open(node_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node"] + node_attr_keys)

        for node, data in sorted(G.nodes(data=True)):
            new_id = new_map[node]
            row = [new_id] + [data.get(k, "") for k in node_attr_keys]
            writer.writerow(row)

    # ---- Collect edge attributes ----
    edge_attr_keys = set()
    for _, _, data in G.edges(data=True):
        edge_attr_keys.update(data.keys())
    edge_attr_keys = sorted(edge_attr_keys)

    # ---- Write edges ----
    with open(edge_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"] + edge_attr_keys)

        for u, v, data in sorted(G.edges(data=True)):
            row = [
                new_map[u],
                new_map[v],
            ] + [data.get(k, "") for k in edge_attr_keys]
            writer.writerow(row)

    return new_map  # useful for debugging / reuse


# ============================================================
from pathlib import Path

import pandas as pd
import torch

from dtos import (
    Node,
    Edge,
)


# ============================================================
# LOAD NODES
# ============================================================

def load_nodes(file: str) -> list[Node]:

    print("LOADING NODES ...")

    file = Path(file)

    if not file.exists():
        raise RuntimeError(f"Failed to open file: {file}")

    df = pd.read_csv(file)

    required = ["node_id", "T", "is_leaf", "state_id"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column '{col}' in {file}"
            )

    nodes = [
        Node(
            id=int(row.node_id),
            T=float(row.T),
            is_leaf=bool(row.is_leaf),
            state=int(row.state_id),
        )
        for row in df.itertuples(index=False)
    ]

    print(f"LOADED {len(nodes)} NODES")

    return nodes


# ============================================================
# LOAD EDGES
# ============================================================

def load_edges(file: str) -> list[Edge]:

    print("LOADING EDGES ...")

    file = Path(file)

    if not file.exists():
        raise RuntimeError(f"Failed to open file: {file}")

    df = pd.read_csv(file)

    required = ["parent", "child", "branch_length"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing column '{col}' in {file}"
            )

    edges = [
        Edge(
            u=int(row.parent),
            v=int(row.child),
            length=float(row.branch_length),
        )
        for row in df.itertuples(index=False)
    ]

    print(f"LOADED {len(edges)} EDGES")

    return edges
