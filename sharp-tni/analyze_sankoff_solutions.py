#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


STATE_NAMES = {
    1: "T1",
    2: "L1",
    3: "L2",
    4: "L3",
    5: "S1",
}
SOLUTION_HEADER = re.compile(
    rb"^# idx = (\d+) -- root = (\d+) -- gamma = (\d+)\s*$"
)
OUTPUT_COLUMNS = (
    "solution_rank_by_sharptni_gamma",
    "solution_index",
    "sharptni_gamma",
    "group_index_by_time",
    "overlap_start_sharptni",
    "overlap_end_sharptni",
    "grouped_time_sharptni",
    "grouped_time_carta",
    "changed_edges_in_group",
    "edge_directions_and_counts",
)

EdgeInterval = tuple[float, float, int, int, int, int]


def _read_solution_header(line: bytes, line_number: int) -> tuple[int, int, int]:
    match = SOLUTION_HEADER.match(line.rstrip(b"\r\n"))
    if match is None:
        raise ValueError(f"Invalid solution header at compressed line {line_number}")
    return tuple(int(value) for value in match.groups())


def _read_nodes(
    source: gzip.GzipFile,
    solution_index: int,
    node_count: int,
) -> list[tuple[float, int, int, int]]:
    nodes = []
    for node_id in range(1, node_count + 1):
        line = source.readline()
        if not line:
            raise EOFError(
                f"Truncated solution {solution_index} at node record {node_id}"
            )
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"Invalid node record {node_id} in solution {solution_index}"
            )
        try:
            time = float(fields[0])
            child1, child2, state = map(int, fields[1:])
        except ValueError as error:
            raise ValueError(
                f"Invalid node record {node_id} in solution {solution_index}"
            ) from error
        if not math.isfinite(time):
            raise ValueError(
                f"Non-finite node time in solution {solution_index}, node {node_id}"
            )
        if not 1 <= state <= len(STATE_NAMES):
            raise ValueError(
                f"Unknown state ID {state} in solution {solution_index}, node {node_id}"
            )
        if not 0 <= child1 <= node_count or not 0 <= child2 <= node_count:
            raise ValueError(
                f"Invalid child ID in solution {solution_index}, node {node_id}"
            )
        nodes.append((time, child1, child2, state))
    return nodes


def _changed_edge_intervals(
    nodes: list[tuple[float, int, int, int]],
    solution_index: int,
) -> list[EdgeInterval]:
    intervals: list[EdgeInterval] = []
    for parent_id, (parent_time, child1, child2, parent_state) in enumerate(
        nodes, start=1
    ):
        for child_id in (child1, child2):
            if child_id == 0:
                continue
            child_time, _, _, child_state = nodes[child_id - 1]
            if parent_state == child_state:
                continue
            start = min(parent_time, child_time)
            end = max(parent_time, child_time)
            if not start < end:
                raise ValueError(
                    f"State-changing edge {parent_id}->{child_id} has zero "
                    f"duration in solution {solution_index}"
                )
            intervals.append(
                (start, end, parent_id, child_id, parent_state, child_state)
            )
    return intervals


def _group_by_positive_overlap(
    intervals: list[EdgeInterval],
) -> list[tuple[float, float, float, list[EdgeInterval]]]:
    """Find a minimum interval cover, allowing mixed transition directions."""
    remaining = sorted(intervals, key=lambda item: (item[1], item[0], item[2], item[3]))
    groups = []
    while remaining:
        earliest_end = remaining[0][1]
        members = [
            interval
            for interval in remaining
            if interval[0] < earliest_end <= interval[1]
        ]
        if not members:
            raise ValueError("Could not form a positive-overlap interval group")

        overlap_start = max(interval[0] for interval in members)
        overlap_end = min(interval[1] for interval in members)
        if not overlap_start < overlap_end:
            raise ValueError("An interval group has no positive-width overlap")
        midpoint = (overlap_start + overlap_end) / 2
        groups.append((overlap_start, overlap_end, midpoint, members))

        grouped_edges = {(item[2], item[3]) for item in members}
        remaining = [
            item for item in remaining if (item[2], item[3]) not in grouped_edges
        ]
    return groups


def _format_group_rows(
    solution_rank: int,
    solution_index: int,
    gamma: int,
    groups: list[tuple[float, float, float, list[EdgeInterval]]],
) -> list[dict[str, object]]:
    rows = []
    for group_index, (start, end, midpoint, members) in enumerate(groups, start=1):
        direction_counts = Counter(
            f"{STATE_NAMES[from_state]}->{STATE_NAMES[to_state]}"
            for _, _, _, _, from_state, to_state in members
        )
        rows.append(
            {
                "solution_rank_by_sharptni_gamma": solution_rank,
                "solution_index": solution_index,
                "sharptni_gamma": gamma,
                "group_index_by_time": group_index,
                "overlap_start_sharptni": f"{start:.12g}",
                "overlap_end_sharptni": f"{end:.12g}",
                "grouped_time_sharptni": f"{midpoint:.12g}",
                "grouped_time_carta": f"{1.0 - midpoint:.12g}",
                "changed_edges_in_group": len(members),
                "edge_directions_and_counts": ";".join(
                    f"{direction}:{count}"
                    for direction, count in sorted(direction_counts.items())
                ),
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def analyze_solutions(
    solutions_path: Path,
    output_dir: Path,
    node_count: int,
    gammas: list[int],
    overwrite: bool = False,
) -> dict[int, int]:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    if not gammas or any(gamma < 0 for gamma in gammas):
        raise ValueError("At least one non-negative gamma value is required")
    requested_gammas = sorted(set(gammas))
    output_paths = {
        gamma: output_dir / f"gamma{gamma}_mixed_direction_transition_times.tsv"
        for gamma in requested_gammas
    }
    if not overwrite:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(str(path) for path in existing)
                + ". Pass --overwrite to replace it."
            )

    rows_by_gamma: dict[int, list[dict[str, object]]] = defaultdict(list)
    ranks_by_gamma: Counter[int] = Counter()
    previous_gamma = -1
    line_number = 0
    max_gamma = requested_gammas[-1]

    with gzip.open(solutions_path, "rb") as source:
        while True:
            header = source.readline()
            if not header:
                break
            line_number += 1
            solution_index, _root_state, gamma = _read_solution_header(
                header, line_number
            )
            if gamma < previous_gamma:
                raise ValueError(
                    "The solution file must be sorted by nondecreasing gamma."
                )
            previous_gamma = gamma
            if gamma > max_gamma:
                break
            if gamma in requested_gammas:
                nodes = _read_nodes(source, solution_index, node_count)
                intervals = _changed_edge_intervals(nodes, solution_index)
                groups = _group_by_positive_overlap(intervals)
                grouped_edge_count = sum(len(group[3]) for group in groups)
                if grouped_edge_count != len(intervals):
                    raise ValueError(
                        f"Grouping did not cover every changed edge in "
                        f"solution {solution_index}"
                    )
                ranks_by_gamma[gamma] += 1
                rows_by_gamma[gamma].extend(
                    _format_group_rows(
                        ranks_by_gamma[gamma],
                        solution_index,
                        gamma,
                        groups,
                    )
                )
            else:
                for _ in range(node_count):
                    if not source.readline():
                        raise EOFError(
                            f"Truncated solution {solution_index} at node records"
                        )

    missing = set(requested_gammas).difference(ranks_by_gamma)
    if missing:
        raise ValueError(f"No solutions found for gamma values: {sorted(missing)}")

    for gamma in requested_gammas:
        _write_rows(output_paths[gamma], rows_by_gamma[gamma], overwrite)
        print(
            f"gamma={gamma}: {ranks_by_gamma[gamma]} solutions, "
            f"{len(rows_by_gamma[gamma])} groups written to {output_paths[gamma]}"
        )
    return dict(ranks_by_gamma)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group state-changing edge intervals across transition directions "
            "for enumerated SharpTNI Sankoff solutions."
        )
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        required=True,
        help="Gzip-compressed Sankoff enumeration sorted by gamma.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for one grouped-transition TSV per requested gamma.",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=2921,
        help="Number of node records following each solution header (default: 2921).",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        nargs="+",
        default=[21, 22],
        help="One or more Sankoff gamma values to analyze (default: 21 22).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output TSVs if they already exist.",
    )
    args = parser.parse_args()

    analyze_solutions(
        args.solutions,
        args.output_dir,
        args.node_count,
        args.gamma,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
