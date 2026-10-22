from __future__ import annotations

import csv
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

from src.plotters import draw_ultrametric_tree


STATE_ID_BY_NAME = {"NC": 0, "PT": 1, "LN": 2}
STATE_NAME_BY_PREFIX = {"gray": "NC", "blue": "PT", "orange": "LN"}
NODE_COLUMNS = ("node_id", "T", "is_leaf", "state_id")
EDGE_COLUMNS = ("parent", "child", "branch_length")
TRANSITION_COLUMNS = (
    "parent",
    "child",
    "state_0",
    "state_1",
    "interval_start",
    "interval_end",
    "transition_time",
)
TRANSITION_TIME_OFFSET = 1e-6
TRANSITION_TIME_DECREMENT = 1e-4


@dataclass(frozen=True)
class TransitionEvent:
    parent: int
    child: int
    state_0: str
    state_1: str
    interval_start: float
    interval_end: float
    transition_time: float


@dataclass
class TreeRepresentation:
    tree: Tree
    graph: nx.DiGraph
    root: int
    clades: tuple[Clade, ...]
    node_id_by_clade_id: dict[int, int]
    clade_by_node_id: dict[int, Clade]
    depths: dict[int, float]
    root_age: float


def load_tree(path: Path) -> Tree:
    if not path.is_file():
        raise FileNotFoundError(f"Input tree does not exist: {path}")
    tree = Phylo.read(path, "newick")
    _remove_zero_root_branch(tree)
    return tree


def build_tree_representation(tree: Tree) -> TreeRepresentation:
    _remove_zero_root_branch(tree)
    clades = tuple(tree.find_clades(order="preorder"))
    node_id_by_clade_id = {
        id(clade): node_id for node_id, clade in enumerate(clades)
    }
    clade_by_node_id = {
        node_id: clade for node_id, clade in enumerate(clades)
    }
    root = node_id_by_clade_id[id(tree.root)]

    depths: dict[int, float] = {root: 0.0}
    graph = nx.DiGraph()
    for clade in clades:
        node = node_id_by_clade_id[id(clade)]
        if node not in depths:
            raise ValueError(f"Tree node {node} is unreachable from the root.")
        graph.add_node(node, label=clade.name or "", terminal=clade.is_terminal())
        for child in clade.clades:
            length = child.branch_length
            if length is None or not math.isfinite(length) or length < 0.0:
                raise ValueError(
                    f"Invalid branch length below node {clade.name!r}: {length}"
                )
            child_node = node_id_by_clade_id[id(child)]
            depths[child_node] = depths[node] + float(length)
            graph.add_edge(node, child_node, length=float(length))

    if graph.number_of_edges() != graph.number_of_nodes() - 1:
        raise ValueError("The input must be a rooted tree without extra edges.")
    if graph.in_degree(root) != 0:
        raise ValueError("The input tree root must not have a parent edge.")

    state_by_clade_id = _infer_clade_states(clades)
    leaf_depths = [
        depths[node_id_by_clade_id[id(clade)]]
        for clade in clades
        if clade.is_terminal()
    ]
    if not leaf_depths:
        raise ValueError("The input tree has no terminal nodes.")
    root_age = max(leaf_depths)

    for clade in clades:
        node = node_id_by_clade_id[id(clade)]
        state = state_by_clade_id[id(clade)]
        graph.nodes[node].update(
            {
                "state": state,
                "is_leaf": clade.is_terminal(),
                "state_id": STATE_ID_BY_NAME.get(state, -1),
                "T": root_age - depths[node],
            }
        )

    if any(data["label"].lower() == "stem" for _, data in graph.nodes(data=True)):
        raise ValueError("A node labeled 'stem' is not allowed in the processed tree.")

    return TreeRepresentation(
        tree=tree,
        graph=graph,
        root=root,
        clades=clades,
        node_id_by_clade_id=node_id_by_clade_id,
        clade_by_node_id=clade_by_node_id,
        depths=depths,
        root_age=root_age,
    )


def export_stage(
    stage: TreeRepresentation,
    output_dir: Path,
    transitions: tuple[TransitionEvent, ...] | list[TransitionEvent] = (),
    title: str = "SGOOTR tree",
    transition_time_scale: float = 1.0,
) -> dict[str, Path]:
    if not math.isfinite(transition_time_scale) or transition_time_scale <= 0.0:
        raise ValueError("transition_time_scale must be finite and positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tree": output_dir / "tree.nwk",
        "nodes": output_dir / "nodes.csv",
        "edges": output_dir / "edges.csv",
        "transition_times": output_dir / "transition_times.csv",
        "transition_times_text": output_dir / "transition_times.txt",
        "plot": output_dir / "tree.png",
    }

    _remove_zero_root_branch(stage.tree)
    Phylo.write(
        stage.tree,
        paths["tree"],
        "newick",
        format_branch_length="%1.10f",
    )
    _remove_serialized_root_branch(paths["tree"])
    _write_nodes(stage, paths["nodes"])
    _write_edges(stage, paths["edges"])
    _write_transitions(
        stage,
        transitions,
        paths["transition_times"],
        transition_time_scale,
    )
    _write_transition_times(
        stage,
        transitions,
        paths["transition_times_text"],
        transition_time_scale,
    )
    _plot_stage(stage, transitions, paths["plot"], title)
    return paths


def _remove_zero_root_branch(tree: Tree) -> None:
    branch_length = tree.root.branch_length
    if branch_length is not None and not math.isclose(
        float(branch_length), 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Root has a nonzero branch length ({branch_length}); "
            "the pipeline does not add or retain stem edges."
        )
    tree.root.branch_length = None


def _remove_serialized_root_branch(path: Path) -> None:
    serialized = path.read_text()
    terminator = serialized.rfind(";")
    if terminator == -1:
        raise ValueError(f"Newick output has no tree terminator: {path}")
    body = serialized[:terminator]
    colon = body.rfind(":")
    if colon == -1:
        return
    branch_and_comments = body[colon + 1 :]
    match = re.match(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        branch_and_comments,
    )
    if match is None:
        return
    suffix = branch_and_comments[match.end() :]
    if suffix and not suffix.startswith("["):
        return
    path.write_text(
        body[:colon] + suffix + serialized[terminator:]
    )


def _infer_clade_states(clades: tuple[Clade, ...]) -> dict[int, str | None]:
    states: dict[int, str | None] = {}
    for clade in reversed(clades):
        if clade.is_terminal():
            label_prefix = (clade.name or "").split("_", maxsplit=1)[0]
            state = STATE_NAME_BY_PREFIX.get(label_prefix)
            if state is None:
                raise ValueError(
                    f"Cannot infer state from terminal label {clade.name!r}; "
                    f"expected one of {tuple(STATE_NAME_BY_PREFIX)}."
                )
            states[id(clade)] = state
            continue

        child_states = {states[id(child)] for child in clade.clades}
        states[id(clade)] = (
            next(iter(child_states)) if len(child_states) == 1 else None
        )
    return states


def _write_nodes(stage: TreeRepresentation, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NODE_COLUMNS)
        for node, data in stage.graph.nodes(data=True):
            writer.writerow(
                [
                    node,
                    f"{float(data['T']):.15g}",
                    int(data["is_leaf"]),
                    int(data["state_id"]),
                ]
            )


def _write_edges(stage: TreeRepresentation, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EDGE_COLUMNS)
        for parent, child, data in stage.graph.edges(data=True):
            writer.writerow(
                [parent, child, f"{float(data['length']):.15g}"]
            )


def _write_transitions(
    stage: TreeRepresentation,
    transitions: tuple[TransitionEvent, ...] | list[TransitionEvent],
    path: Path,
    transition_time_scale: float,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRANSITION_COLUMNS)
        for event in transitions:
            interval_start, interval_end = sorted(
                (
                    _inverse_transition_time(
                        event.interval_start,
                        stage.root_age,
                        transition_time_scale,
                    ),
                    _inverse_transition_time(
                        event.interval_end,
                        stage.root_age,
                        transition_time_scale,
                    ),
                )
            )
            writer.writerow(
                [
                    event.parent,
                    event.child,
                    event.state_0,
                    event.state_1,
                    f"{interval_start:.15g}",
                    f"{interval_end:.15g}",
                    (
                        f"{_export_transition_time(event.transition_time, stage.root_age, transition_time_scale):.15g}"
                    ),
                ]
            )


def _write_transition_times(
    stage: TreeRepresentation,
    transitions: tuple[TransitionEvent, ...] | list[TransitionEvent],
    path: Path,
    transition_time_scale: float,
) -> None:
    times = sorted(
        {
            _export_transition_time(
                event.transition_time,
                stage.root_age,
                transition_time_scale,
            )
            for event in transitions
        }
    )
    path.write_text("".join(f"{time:.15g}\n" for time in times))


def _inverse_transition_time(
    time: float,
    maximum_time: float,
    time_scale: float = 1.0,
) -> float:
    return maximum_time - time + TRANSITION_TIME_OFFSET * time_scale


def _export_transition_time(
    time: float,
    maximum_time: float,
    time_scale: float = 1.0,
) -> float:
    return (
        _inverse_transition_time(time, maximum_time, time_scale)
        - TRANSITION_TIME_DECREMENT * time_scale
    )


def _plot_stage(
    stage: TreeRepresentation,
    transitions: tuple[TransitionEvent, ...] | list[TransitionEvent],
    path: Path,
    title: str,
) -> None:
    node_colors = [
        (node, data["state"])
        for node, data in stage.graph.nodes(data=True)
        if data["state"] is not None
    ]
    plot_transitions = [
        (
            event.parent,
            event.child,
            event.state_0,
            event.state_1,
            stage.root_age - event.transition_time,
        )
        for event in transitions
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        draw_ultrametric_tree(
            stage.graph,
            root=stage.root,
            add_labels=False,
            figsize=(20, 6),
            node_colors=node_colors,
            transition_times_of_edges=plot_transitions,
            diff_times=sorted({event.transition_time for event in transitions}),
        )
    axis = plt.gca()
    axis.set_title(title)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
