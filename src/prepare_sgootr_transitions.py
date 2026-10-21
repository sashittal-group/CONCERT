from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


STATE_IDS = {"NC": 0, "PT": 1, "LN": 2}
TIME_EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SGOOTR/MACH2 transitions to per-edge inference inputs."
    )
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-times-output", type=Path, required=True)
    return parser.parse_args()


def read_nodes(path: Path) -> dict[int, float]:
    with path.open(newline="") as node_file:
        reader = csv.DictReader(node_file)
        if not {"node_id", "T"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain node_id and T columns.")
        return {int(row["node_id"]): float(row["T"]) for row in reader}


def read_edges(path: Path) -> set[tuple[int, int]]:
    with path.open(newline="") as edge_file:
        reader = csv.DictReader(edge_file)
        if not {"parent", "child"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain parent and child columns.")
        return {(int(row["parent"]), int(row["child"])) for row in reader}


def convert_transitions(
    transitions_path: Path,
    nodes_path: Path,
    edges_path: Path,
) -> tuple[list[dict[str, int | float]], list[float]]:
    node_times = read_nodes(nodes_path)
    tree_edges = read_edges(edges_path)
    roots = set(node_times).difference(child for _, child in tree_edges)
    if len(roots) != 1:
        raise ValueError(
            f"Expected one root in {nodes_path} and {edges_path}; found {len(roots)}."
        )
    inference_root = roots.pop()
    required_columns = {
        "parent",
        "child",
        "state_0",
        "state_1",
        "interval_start",
        "interval_end",
        "transition_time",
    }

    converted = []
    with transitions_path.open(newline="") as transition_file:
        reader = csv.DictReader(transition_file)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{transitions_path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                # MACH2 uses one-based origin IDs and -1 for the synthetic root parent.
                parent = int(row["parent"]) - 1
                child = int(row["child"]) - 1
                if parent == -1:
                    parent = inference_root
                state_0 = STATE_IDS[row["state_0"]]
                state_1 = STATE_IDS[row["state_1"]]
                interval_start = float(row["interval_start"])
                interval_end = float(row["interval_end"])
                source_time = float(row["transition_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid transition at {transitions_path}:{line_number}: {row}"
                ) from error

            if not all(
                math.isfinite(value)
                for value in (interval_start, interval_end, source_time)
            ):
                raise ValueError(
                    f"Non-finite transition time at {transitions_path}:{line_number}."
                )
            if not interval_start < interval_end:
                raise ValueError(
                    f"Invalid branch interval at {transitions_path}:{line_number}."
                )
            if not interval_start <= source_time <= interval_end:
                raise ValueError(
                    f"Transition time is outside its branch interval at "
                    f"{transitions_path}:{line_number}."
                )
            if (parent, child) not in tree_edges:
                raise ValueError(
                    f"Mapped transition edge {parent}->{child} from "
                    f"{transitions_path}:{line_number} is absent from {edges_path}."
                )
            if parent not in node_times or child not in node_times:
                raise ValueError(
                    f"Transition edge {parent}->{child} has no node times."
                )

            parent_time = node_times[parent]
            child_time = node_times[child]
            branch_duration = parent_time - child_time
            source_duration = interval_end - interval_start
            if branch_duration <= 0 or not math.isclose(
                branch_duration,
                source_duration,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Branch interval at {transitions_path}:{line_number} does not "
                    f"match edge {parent}->{child} times."
                )

            fraction = (source_time - interval_start) / source_duration
            event_time = parent_time - fraction * branch_duration - TIME_EPSILON
            if not child_time < event_time < parent_time:
                raise ValueError(
                    f"Transition at {transitions_path}:{line_number} is too close "
                    "to a branch endpoint for the configured time adjustment."
                )

            converted.append(
                {
                    "parent": parent,
                    "child": child,
                    "s0": state_0,
                    "s1": state_1,
                    "t": event_time,
                }
            )

    if not converted:
        raise ValueError(f"No per-edge transitions found in {transitions_path}.")

    event_times = sorted(row["t"] for row in converted)
    epochs = []
    for event_time in event_times:
        if not epochs or not math.isclose(
            event_time, epochs[-1], rel_tol=1e-9, abs_tol=1e-12
        ):
            epochs.append(event_time)
    for row in converted:
        row["t"] = min(epochs, key=lambda epoch: abs(epoch - row["t"]))

    return converted, epochs


def main() -> None:
    args = parse_args()
    transitions, transition_times = convert_transitions(
        args.transitions,
        args.nodes,
        args.edges,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["parent", "child", "s0", "s1", "t"],
        )
        writer.writeheader()
        writer.writerows(transitions)

    args.transition_times_output.parent.mkdir(parents=True, exist_ok=True)
    with args.transition_times_output.open("w") as time_file:
        for transition_time in transition_times:
            time_file.write(f"{transition_time:.15g}\n")

    print(
        f"Wrote {len(transitions)} per-edge transitions across "
        f"{len(transition_times)} epochs to {args.output}."
    )


if __name__ == "__main__":
    main()
