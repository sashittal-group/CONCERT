from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.analyses.common.executables import resolve_executable
from src.concert.state_ids import STATE_ID_BY_NAME
from src.analyses.crc11.preprocessing.tree import (
    TransitionEvent,
    TreeRepresentation,
)


STATE_NAME_BY_ID = {state_id: state for state, state_id in STATE_ID_BY_NAME.items()}


def infer_transition_events(
    stage: TreeRepresentation,
    output_dir: Path,
    project_root: Path,
    *,
    executable: str | None = None,
) -> list[TransitionEvent]:
    mach2 = resolve_executable("mach2", executable, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    mach2_id_by_node = {node: node + 1 for node in stage.graph.nodes}
    node_by_mach2_id = {
        mach2_id: node for node, mach2_id in mach2_id_by_node.items()
    }
    mach2_edges = [
        (mach2_id_by_node[parent], mach2_id_by_node[child])
        for parent, child in stage.graph.edges
    ]
    tip_labels = [
        (mach2_id_by_node[node], STATE_ID_BY_NAME[data["state"]])
        for node, data in stage.graph.nodes(data=True)
        if data["terminal"]
    ]
    if len(tip_labels) != sum(data["terminal"] for _, data in stage.graph.nodes(data=True)):
        raise ValueError("Every terminal node must have a recognized state label.")

    with tempfile.TemporaryDirectory(prefix="sgootr-mach2-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        tree_path = temporary_path / "tree.tsv"
        labeling_path = temporary_path / "labeling.tsv"
        mach2_output = temporary_path / "output"
        mach2_output.mkdir()
        tree_path.write_text(
            "".join(f"{parent}\t{child}\n" for parent, child in mach2_edges)
        )
        labeling_path.write_text(
            "".join(f"{node}\t{state}\n" for node, state in tip_labels)
        )

        command = [
            str(mach2),
            str(tree_path),
            str(labeling_path),
            "-p",
            "0",
            "-o",
            str(mach2_output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path = output_dir / "mach2.log"
        log_path.write_text(
            f"$ {' '.join(command)}\n\n"
            f"Exit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MACH2 failed with exit code {result.returncode}; see {log_path}."
            )

        refined_paths = sorted(mach2_output.glob("*-T-0.refined.tree"))
        if len(refined_paths) != 1:
            raise FileNotFoundError(
                f"Expected one first-solution MACH2 tree, found {len(refined_paths)}."
            )
        refined_path = refined_paths[0]
        prefix = str(refined_path)[: -len(".refined.tree")]
        refined = pd.read_csv(
            refined_path,
            sep="\t",
            header=None,
            names=("parent", "child", "mach2_time"),
            dtype={"parent": str, "child": str},
        )
        origins = pd.read_csv(
            f"{prefix}.nodes",
            sep="\t",
            header=None,
            names=("node_id", "origin_node_id"),
            dtype=str,
        )
        locations = pd.read_csv(
            f"{prefix}.location.labeling",
            sep="\t",
            header=None,
            names=("node_id", "state_id"),
            dtype={"node_id": str, "state_id": int},
        )

    origin_by_node = dict(zip(origins["node_id"], origins["origin_node_id"]))
    state_by_node = dict(zip(locations["node_id"], locations["state_id"]))
    time_by_mach2_id = {
        str(mach2_id_by_node[node]): stage.depths[node]
        for node in stage.graph.nodes
    }
    migration_edges = refined.loc[refined["mach2_time"] != -1].copy()
    if migration_edges.empty:
        raise ValueError("MACH2 did not infer any state transitions.")

    transition_time_by_edge = _group_overlapping_transition_times(
        migration_edges,
        origin_by_node,
        state_by_node,
        time_by_mach2_id,
    )
    events = []
    for edge_index, edge in migration_edges.iterrows():
        parent_origin = origin_by_node[str(edge["parent"])]
        child_origin = origin_by_node[str(edge["child"])]
        parent_mach2_id = int(parent_origin)
        child_mach2_id = int(child_origin)
        if parent_mach2_id not in node_by_mach2_id or child_mach2_id not in node_by_mach2_id:
            raise ValueError(
                f"MACH2 transition edge {edge['parent']}->{edge['child']} "
                "does not map to input tree nodes."
            )
        parent = node_by_mach2_id[parent_mach2_id]
        child = node_by_mach2_id[child_mach2_id]
        if not stage.graph.has_edge(parent, child):
            raise ValueError(
                f"MACH2 transition edge {parent}->{child} is not an edge "
                "in the input tree."
            )

        source_state_id = state_by_node[str(edge["parent"])]
        target_state_id = state_by_node[str(edge["child"])]
        if source_state_id not in STATE_NAME_BY_ID or target_state_id not in STATE_NAME_BY_ID:
            raise ValueError(
                f"MACH2 returned unknown state IDs {source_state_id}->{target_state_id}."
            )
        start, end = sorted(
            (stage.depths[parent], stage.depths[child])
        )
        transition_time = transition_time_by_edge[edge_index]
        if not start - 1e-9 <= transition_time <= end + 1e-9:
            raise ValueError(
                f"Transition time {transition_time:.10g} falls outside edge "
                f"{parent}->{child} [{start:.10g}, {end:.10g}]."
            )
        events.append(
            TransitionEvent(
                parent=parent,
                child=child,
                state_0=STATE_NAME_BY_ID[source_state_id],
                state_1=STATE_NAME_BY_ID[target_state_id],
                interval_start=start,
                interval_end=end,
                transition_time=transition_time,
            )
        )
    return events


def project_transition_events(
    source: TreeRepresentation,
    target: TreeRepresentation,
    events: list[TransitionEvent],
) -> list[TransitionEvent]:
    projected = []
    for event in events:
        if not source.graph.has_edge(event.parent, event.child):
            raise ValueError(
                f"Transition edge {event.parent}->{event.child} is absent "
                "from the source tree."
            )
        if not target.graph.has_edge(event.parent, event.child):
            raise ValueError(
                f"Transition edge {event.parent}->{event.child} is absent "
                "from the target tree."
            )

        source_start = source.depths[event.parent]
        source_end = source.depths[event.child]
        source_duration = source_end - source_start
        if source_duration <= 1e-12:
            raise ValueError(
                f"Cannot project a transition across zero-length edge "
                f"{event.parent}->{event.child}."
            )
        fraction = (event.transition_time - source_start) / source_duration
        if not -1e-8 <= fraction <= 1.0 + 1e-8:
            raise ValueError(
                f"Transition time {event.transition_time:.10g} is outside "
                f"edge {event.parent}->{event.child}."
            )
        fraction = min(1.0, max(0.0, fraction))
        target_start = target.depths[event.parent]
        target_end = target.depths[event.child]
        target_time = target_start + fraction * (target_end - target_start)
        projected.append(
            TransitionEvent(
                parent=event.parent,
                child=event.child,
                state_0=event.state_0,
                state_1=event.state_1,
                interval_start=min(target_start, target_end),
                interval_end=max(target_start, target_end),
                transition_time=target_time,
            )
        )
    return projected


def retain_transition_times(
    stage: TreeRepresentation,
    events: list[TransitionEvent],
) -> list[TransitionEvent]:
    retained = []
    for event in events:
        if not stage.graph.has_edge(event.parent, event.child):
            raise ValueError(
                f"Transition edge {event.parent}->{event.child} is absent "
                "from the target tree."
            )
        start = stage.depths[event.parent]
        end = stage.depths[event.child]
        if not min(start, end) - 1e-9 <= event.transition_time <= max(start, end) + 1e-9:
            raise ValueError(
                f"Fixed transition time {event.transition_time:.10g} no longer "
                f"lies on edge {event.parent}->{event.child}."
            )
        retained.append(
            TransitionEvent(
                parent=event.parent,
                child=event.child,
                state_0=event.state_0,
                state_1=event.state_1,
                interval_start=min(start, end),
                interval_end=max(start, end),
                transition_time=event.transition_time,
            )
        )
    return retained


def _group_overlapping_transition_times(
    migration_edges: pd.DataFrame,
    origin_by_node: dict[str, str],
    state_by_node: dict[str, int],
    time_by_mach2_id: dict[str, float],
) -> dict[int, float]:
    grouped_edges: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    for edge_index, edge in migration_edges.iterrows():
        parent_origin = origin_by_node[str(edge["parent"])]
        child_origin = origin_by_node[str(edge["child"])]
        source_state = state_by_node[str(edge["parent"])]
        target_state = state_by_node[str(edge["child"])]
        if source_state not in STATE_NAME_BY_ID or target_state not in STATE_NAME_BY_ID:
            raise ValueError(
                f"MACH2 returned unknown state IDs {source_state}->{target_state}."
            )
        start = time_by_mach2_id[parent_origin]
        end = time_by_mach2_id[child_origin]
        grouped_edges[(source_state, target_state)].append(
            {
                "edge_index": int(edge_index),
                "start": min(start, end),
                "end": max(start, end),
            }
        )

    transition_time_by_edge: dict[int, float] = {}
    for edges in grouped_edges.values():
        edges.sort(key=lambda item: float(item["end"]))
        overlap_group: list[dict[str, float | int]] = []
        shared_start = 0.0
        shared_end = 0.0
        for edge in edges:
            start = float(edge["start"])
            end = float(edge["end"])
            if overlap_group:
                next_start = max(shared_start, start)
                next_end = min(shared_end, end)
            else:
                next_start = start
                next_end = end

            if overlap_group and next_start < next_end:
                overlap_group.append(edge)
                shared_start = next_start
                shared_end = next_end
            else:
                _assign_group_time(
                    transition_time_by_edge,
                    overlap_group,
                    shared_start,
                )
                overlap_group = [edge]
                shared_start = start
                shared_end = end
        _assign_group_time(
            transition_time_by_edge,
            overlap_group,
            shared_start,
        )

    return transition_time_by_edge


def _assign_group_time(
    transition_time_by_edge: dict[int, float],
    edges: list[dict[str, float | int]],
    shared_start: float,
) -> None:
    for edge in edges:
        transition_time_by_edge[int(edge["edge_index"])] = shared_start
