from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def process_transition_events(
    graph: nx.DiGraph,
    refined_tree_path: Path,
    output_dir: Path,
    *,
    minimum_edge_length: float = 0.001,
) -> dict[str, Path]:
    if minimum_edge_length <= 0.0:
        raise ValueError("minimum_edge_length must be positive")

    edges = pd.read_csv(refined_tree_path, sep="\t", header=None)
    if edges.shape[1] != 3:
        raise ValueError(
            f"Expected three columns in refined tree {refined_tree_path}"
        )
    edges.columns = ["parent", "child", "edge_class"]
    edges["edge_index"] = range(len(edges))
    edges = edges.loc[edges["edge_class"] != -1].copy()
    root = graph.graph.get("root")
    if root is not None:
        edges = edges.loc[
            ~((edges["parent"] == 0) & (edges["child"] == root))
        ].copy()

    times = nx.get_node_attributes(graph, "T")
    states = nx.get_node_attributes(graph, "state")
    if edges[["parent", "child"]].isna().any().any():
        raise ValueError(f"Missing endpoint IDs in {refined_tree_path}")
    edges["parent"] = edges["parent"].astype(int)
    edges["child"] = edges["child"].astype(int)
    edges["T_0"] = edges["parent"].map(times)
    edges["T_1"] = edges["child"].map(times)
    edges["state_0"] = edges["parent"].map(states)
    edges["state_1"] = edges["child"].map(states)
    if edges[["T_0", "T_1", "state_0", "state_1"]].isna().any().any():
        raise ValueError(
            f"Refined tree {refined_tree_path} references nodes without time or state"
        )

    edges["T_0"] = edges["T_0"].astype(float)
    edges["T_1"] = edges["T_1"].astype(float)
    edges["state_0"] = edges["state_0"].astype(int)
    edges["state_1"] = edges["state_1"].astype(int)
    raw_transitions = edges[
        ["parent", "child", "state_0", "state_1"]
    ].copy()
    raw_transitions["t"] = (edges["T_0"] + edges["T_1"]) / 2.0

    transition_edges = edges.loc[edges["state_0"] != edges["state_1"]].copy()
    transition_edges["interval_start"] = transition_edges[["T_0", "T_1"]].min(axis=1)
    transition_edges["interval_end"] = transition_edges[["T_0", "T_1"]].max(axis=1)

    transition_time_by_edge, transition_times = _group_transition_intervals(
        transition_edges,
        minimum_edge_length / 2.0,
    )
    adjusted_transitions = transition_edges[
        ["edge_index", "parent", "child", "state_0", "state_1"]
    ].copy()
    adjusted_transitions["t"] = adjusted_transitions["edge_index"].map(
        transition_time_by_edge
    )
    adjusted_transitions = adjusted_transitions[
        ["parent", "child", "state_0", "state_1", "t"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "transitions_0.csv"
    adjusted_path = output_dir / "transitions_0_adjusted.csv"
    times_path = output_dir / "transition-times.txt"
    raw_transitions.to_csv(raw_path, index=False)
    adjusted_transitions.to_csv(adjusted_path, index=False)
    times_path.write_text("".join(f"{time}\n" for time in transition_times))
    return {
        "transitions": raw_path,
        "adjusted_transitions": adjusted_path,
        "transition_times": times_path,
    }


def _group_transition_intervals(
    transitions: pd.DataFrame,
    zero_time_offset: float,
) -> tuple[dict[int, float], list[float]]:
    time_by_edge: dict[int, float] = {}
    group_times: list[float] = []

    for _, state_edges in transitions.groupby(
        ["state_0", "state_1"],
        sort=False,
    ):
        ordered_edges = state_edges.sort_values("interval_end", kind="stable")
        active_indices: list[int] = []
        shared_start: float | None = None
        shared_end: float | None = None

        for edge in ordered_edges.itertuples(index=False):
            edge_start = float(edge.interval_start)
            edge_end = float(edge.interval_end)
            if not active_indices:
                active_indices = [int(edge.edge_index)]
                shared_start = edge_start
                shared_end = edge_end
                continue

            next_start = max(shared_start, edge_start)
            next_end = min(shared_end, edge_end)
            if next_start < next_end:
                active_indices.append(int(edge.edge_index))
                shared_start = next_start
                shared_end = next_end
                continue

            _record_transition_group(
                active_indices,
                shared_start,
                zero_time_offset,
                time_by_edge,
                group_times,
            )
            active_indices = [int(edge.edge_index)]
            shared_start = edge_start
            shared_end = edge_end

        if active_indices:
            _record_transition_group(
                active_indices,
                shared_start,
                zero_time_offset,
                time_by_edge,
                group_times,
            )

    return time_by_edge, sorted(set(group_times))


def _record_transition_group(
    edge_indices: list[int],
    start: float | None,
    zero_time_offset: float,
    time_by_edge: dict[int, float],
    group_times: list[float],
) -> None:
    if start is None:
        raise ValueError("Transition group is missing its shared interval start")
    event_time = zero_time_offset if start == 0.0 else start
    group_times.append(event_time)
    for edge_index in edge_indices:
        time_by_edge[edge_index] = event_time
