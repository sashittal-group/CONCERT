from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from Bio.Phylo.BaseTree import Clade, Tree

from src.analyses.crc11.preprocessing.tree import (
    TRANSITION_TIME_DECREMENT,
    TRANSITION_TIME_OFFSET,
    TransitionEvent,
    TreeRepresentation,
    build_tree_representation,
    export_stage,
    load_tree,
)
from src.analyses.crc11.preprocessing.upgma import (
    build_upgma_tree,
    scale_tree_to_height,
    subtract_minimum_pairwise_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sgootr"
    / "processed"
    / "upgma_min_subtracted_height_1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sgootr"
    / "processed"
    / "upgma_min_subtracted_height_1_transition_cut_upgma"
)
TREE_TOLERANCE = 1e-9


def run_transition_cut_upgma(
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    tree = load_tree(input_dir / "tree.nwk")
    source = build_tree_representation(tree)
    source_tip_names = {tip.name for tip in tree.get_terminals()}
    transitions = _read_transitions(
        input_dir / "transition_times.csv",
        input_dir / "transition_times.txt",
    )
    cutoff = min(event.transition_time for event in transitions)
    if not 0.0 < cutoff < source.root_age:
        raise ValueError(
            f"The latest transition time {cutoff:.15g} must be between "
            f"0 and the root age {source.root_age:.15g}."
        )

    cutoff_depth = source.root_age - cutoff
    if any(
        math.isclose(
            float(data["T"]),
            cutoff,
            rel_tol=0.0,
            abs_tol=TREE_TOLERANCE,
        )
        for _, data in source.graph.nodes(data=True)
    ):
        raise ValueError(
            "The transition cutoff coincides with a tree node; "
            "it must fall inside the intersected branches."
        )

    crossing_edges = [
        (parent, child)
        for parent, child in source.graph.edges
        if source.depths[parent] < cutoff_depth < source.depths[child]
    ]
    if not crossing_edges:
        raise ValueError(
            f"No branches cross the transition cutoff at age {cutoff:.15g}."
        )

    replacement_child_by_edge: dict[tuple[int, int], Clade] = {}
    pairwise_rows = []
    summary_rows = []
    for subtree_id, (parent, child) in enumerate(crossing_edges):
        parent_clade = source.clade_by_node_id[parent]
        child_clade = source.clade_by_node_id[child]
        tips = child_clade.get_terminals()
        if not tips:
            raise ValueError(
                f"Cut subtree {parent}->{child} has no terminal nodes."
            )

        minimum_distance = None
        upgma_height = None
        scale_factor = None
        method = "single_tip_retained"
        replacement_clade = child_clade
        if len(tips) > 1:
            names = tuple(tip.name or "" for tip in tips)
            if not all(names):
                raise ValueError(
                    f"Every terminal in cut subtree {parent}->{child} "
                    "must have a label."
                )
            if len(set(names)) != len(names):
                raise ValueError(
                    f"Terminal labels are not unique in cut subtree "
                    f"{parent}->{child}."
                )
            matrix = _pairwise_distances(tree, tips)
            adjusted_matrix, minimum_distance = subtract_minimum_pairwise_distance(
                names,
                matrix,
            )
            for row in range(len(names)):
                for column in range(row):
                    pairwise_rows.append(
                        {
                            "subtree_id": subtree_id,
                            "tip_1": names[column],
                            "tip_2": names[row],
                            "distance": f"{matrix[row][column]:.15g}",
                            "adjusted_distance": (
                                f"{adjusted_matrix[row][column]:.15g}"
                            ),
                        }
                    )

            subtree_tree = build_upgma_tree(names, adjusted_matrix)
            subtree_stage = build_tree_representation(subtree_tree)
            upgma_height = subtree_stage.root_age
            if upgma_height > 0.0:
                scale_factor = scale_tree_to_height(subtree_tree, cutoff)
                method = "scaled_upgma"
            else:
                for tip in subtree_tree.get_terminals():
                    tip.branch_length = float(tip.branch_length or 0.0) + cutoff
                method = "extended_zero_height_upgma"

            replacement_clade = subtree_tree.root
            replacement_clade.branch_length = (
                cutoff_depth - source.depths[parent]
            )
            parent_clade.clades[parent_clade.clades.index(child_clade)] = (
                replacement_clade
            )
            replacement_child_by_edge[(parent, child)] = replacement_clade

        summary_rows.append(
            {
                "subtree_id": subtree_id,
                "source_parent": parent,
                "source_child": child,
                "tip_count": len(tips),
                "minimum_pairwise_distance": (
                    ""
                    if minimum_distance is None
                    else f"{minimum_distance:.15g}"
                ),
                "upgma_height_before_adjustment": (
                    "" if upgma_height is None else f"{upgma_height:.15g}"
                ),
                "height_scale_factor": (
                    "" if scale_factor is None else f"{scale_factor:.15g}"
                ),
                "adjustment": method,
            }
        )

    stage = build_tree_representation(tree)
    _validate_output_tree(
        source,
        stage,
        cutoff_depth,
        source_tip_names,
    )
    output_events = _map_transition_events(
        source,
        stage,
        transitions,
        replacement_child_by_edge,
        cutoff,
    )
    outputs = export_stage(
        stage,
        output_dir,
        output_events,
        title=f"UPGMA below latest transition at T={cutoff:.4g}",
    )
    outputs["subtree_distances"] = _write_csv(
        output_dir / "subtree_pairwise_distances.csv",
        (
            "subtree_id",
            "tip_1",
            "tip_2",
            "distance",
            "adjusted_distance",
        ),
        pairwise_rows,
    )
    outputs["subtree_summary"] = _write_csv(
        output_dir / "subtree_summary.csv",
        (
            "subtree_id",
            "source_parent",
            "source_child",
            "tip_count",
            "minimum_pairwise_distance",
            "upgma_height_before_adjustment",
            "height_scale_factor",
            "adjustment",
        ),
        summary_rows,
    )
    print(
        f"Kept the tree above T={cutoff:.15g} and processed "
        f"{len(crossing_edges)} descendant subtrees into {output_dir}."
    )
    return outputs


def _pairwise_distances(
    tree: Tree,
    tips: list[Clade],
) -> tuple[tuple[float, ...], ...]:
    matrix = [[0.0] * len(tips) for _ in tips]
    for row, tip in enumerate(tips):
        for column in range(row):
            distance = float(tree.distance(tip, tips[column]))
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    f"Invalid distance between {tips[row].name!r} and "
                    f"{tips[column].name!r}: {distance}"
                )
            matrix[row][column] = distance
            matrix[column][row] = distance
    return tuple(tuple(row) for row in matrix)


def _read_transitions(
    csv_path: Path,
    text_path: Path,
) -> list[TransitionEvent]:
    required_columns = {
        "parent",
        "child",
        "state_0",
        "state_1",
        "interval_start",
        "interval_end",
        "transition_time",
    }
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Transition CSV must contain {sorted(required_columns)}: "
                f"{csv_path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"No per-edge transitions were found in {csv_path}.")

    events = []
    for row_number, row in enumerate(rows, start=2):
        try:
            event = TransitionEvent(
                parent=int(row["parent"]),
                child=int(row["child"]),
                state_0=row["state_0"],
                state_1=row["state_1"],
                interval_start=float(row["interval_start"]),
                interval_end=float(row["interval_end"]),
                transition_time=float(row["transition_time"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid transition data in {csv_path}:{row_number}."
            ) from exc
        if any(
            not math.isfinite(value)
            for value in (
                event.interval_start,
                event.interval_end,
                event.transition_time,
            )
        ):
            raise ValueError(
                f"Non-finite transition data in {csv_path}:{row_number}."
            )
        events.append(event)

    text_times = []
    for line_number, line in enumerate(text_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition time in {text_path}:{line_number}."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite transition time in {text_path}:{line_number}."
            )
        text_times.append(value)

    csv_times = sorted({event.transition_time for event in events})
    if len(text_times) != len(csv_times) or any(
        not math.isclose(text_time, csv_time, rel_tol=0.0, abs_tol=1e-12)
        for text_time, csv_time in zip(text_times, csv_times)
    ):
        raise ValueError(
            f"Per-edge and global transition times disagree between "
            f"{csv_path} and {text_path}."
        )
    return events


def _map_transition_events(
    source: TreeRepresentation,
    target: TreeRepresentation,
    events: list[TransitionEvent],
    replacement_child_by_edge: dict[tuple[int, int], Clade],
    cutoff: float,
) -> list[TransitionEvent]:
    mapped = []
    for event in events:
        edge = (event.parent, event.child)
        if not source.graph.has_edge(*edge):
            raise ValueError(f"Transition edge {edge[0]}->{edge[1]} is absent.")
        if event.transition_time < cutoff - TREE_TOLERANCE:
            raise ValueError(
                f"Transition time {event.transition_time:.15g} is below the "
                f"selected cutoff {cutoff:.15g}."
            )

        parent_clade = source.clade_by_node_id[event.parent]
        child_clade = replacement_child_by_edge.get(
            edge,
            source.clade_by_node_id[event.child],
        )
        try:
            parent = target.node_id_by_clade_id[id(parent_clade)]
            child = target.node_id_by_clade_id[id(child_clade)]
        except KeyError as exc:
            raise ValueError(
                f"Could not map transition edge {edge[0]}->{edge[1]} "
                "onto the transformed tree."
            ) from exc
        if not target.graph.has_edge(parent, child):
            raise ValueError(
                f"Mapped transition edge {parent}->{child} is absent "
                "from the transformed tree."
            )

        transition_depth = (
            target.root_age
            + TRANSITION_TIME_OFFSET
            - TRANSITION_TIME_DECREMENT
            - event.transition_time
        )
        parent_depth = target.depths[parent]
        child_depth = target.depths[child]
        if not min(parent_depth, child_depth) - TREE_TOLERANCE <= (
            transition_depth
        ) <= max(parent_depth, child_depth) + TREE_TOLERANCE:
            raise ValueError(
                f"Transition time {event.transition_time:.15g} does not lie "
                f"on mapped edge {parent}->{child}."
            )
        mapped.append(
            TransitionEvent(
                parent=parent,
                child=child,
                state_0=event.state_0,
                state_1=event.state_1,
                interval_start=min(parent_depth, child_depth),
                interval_end=max(parent_depth, child_depth),
                transition_time=transition_depth,
            )
        )
    return mapped


def _validate_output_tree(
    source: TreeRepresentation,
    target: TreeRepresentation,
    cutoff_depth: float,
    source_tip_names: set[str | None],
) -> None:
    target_tips = {
        clade.name for clade in target.tree.get_terminals()
    }
    if source_tip_names != target_tips:
        raise ValueError("The transformed tree changed the set of terminal labels.")

    tip_depths = [
        target.depths[node]
        for node, data in target.graph.nodes(data=True)
        if data["terminal"]
    ]
    if max(tip_depths) - min(tip_depths) > TREE_TOLERANCE:
        raise ValueError("The transformed tree is not ultrametric.")
    if not math.isclose(
        target.root_age,
        source.root_age,
        rel_tol=0.0,
        abs_tol=TREE_TOLERANCE,
    ):
        raise ValueError(
            f"The transformed tree height {target.root_age:.15g} differs "
            f"from the source height {source.root_age:.15g}."
        )

    for node, depth in source.depths.items():
        if depth >= cutoff_depth:
            continue
        clade = source.clade_by_node_id[node]
        target_node = target.node_id_by_clade_id.get(id(clade))
        if target_node is None:
            raise ValueError(
                f"Source node {node} above the cutoff was not preserved."
            )
        if not math.isclose(
            target.depths[target_node],
            depth,
            rel_tol=0.0,
            abs_tol=TREE_TOLERANCE,
        ):
            raise ValueError(
                f"Source node {node} above the cutoff changed its time."
            )


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the tree above the latest transition and independently "
            "rebuild each descendant subtree with UPGMA."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run_transition_cut_upgma(arguments.input_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
