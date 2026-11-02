#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


STATE_NAMES = ("T1", "L1", "L2", "L3", "S1")


def read_pairs(path: Path) -> dict[int, int]:
    pairs: dict[int, int] = {}
    with path.open(newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Expected two tab-separated columns in {path}")
            node, label = map(int, row)
            if node in pairs:
                raise ValueError(f"Duplicate node {node} in {path}")
            pairs[node] = label
    return pairs


def prepare_inputs(processed_dir: Path, output_dir: Path) -> tuple[int, int]:
    nodes_path = processed_dir / "nodes_0.csv"
    tree_path = processed_dir / "1461.tree"
    leaves_path = processed_dir / "1461.labeling"
    labels_path = processed_dir / "0-T-0.location.labeling"

    for path in (nodes_path, tree_path, leaves_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required KPTracer input not found: {path}\n"
                "Run the KPTracer preprocessing pipeline first."
            )

    with nodes_path.open(newline="") as file:
        node_rows = list(csv.DictReader(file))
    node_times = {
        int(row["node_id"]): float(row["T"])
        for row in node_rows
        if int(row["node_id"]) != 0
    }
    if not node_times or not all(math.isfinite(time) for time in node_times.values()):
        raise ValueError(f"Invalid node times in {nodes_path}")
    earliest = min(node_times.values())
    latest = max(node_times.values())
    if latest <= earliest:
        raise ValueError("KPTracer tree must have positive temporal depth")
    normalized_times = {
        node: (latest - time) / (latest - earliest)
        for node, time in node_times.items()
    }

    children: dict[int, list[int]] = defaultdict(list)
    with tree_path.open(newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Expected two tab-separated columns in {tree_path}")
            parent, child = map(int, row)
            children[parent].append(child)

    node_ids = set(node_times)
    if any(node not in node_ids for pair in children.items() for node in (pair[0], *pair[1])):
        raise ValueError("KPTracer edge list and node table contain different node IDs")
    roots = node_ids.difference(child for descendants in children.values() for child in descendants)
    if len(roots) != 1:
        raise ValueError(f"Expected one KPTracer tree root; found {len(roots)}")
    root = next(iter(roots))

    leaf_labels = read_pairs(leaves_path)
    ancestral_labels = read_pairs(labels_path)
    ancestral_labels.pop(0, None)
    leaves = {node for node in node_ids if not children[node]}
    if set(leaf_labels) != leaves:
        raise ValueError("Tip labels do not match the leaves in the KPTracer tree")
    if any(ancestral_labels.get(node) != label for node, label in leaf_labels.items()):
        raise ValueError("Normalized tree labels disagree with the KPTracer tip labels")
    if set(ancestral_labels) != node_ids:
        raise ValueError("Location labels do not cover the KPTracer tree")
    if any(label < 0 or label >= len(STATE_NAMES) for label in ancestral_labels.values()):
        raise ValueError("KPTracer contains a location label outside the five-state mapping")
    if any(len(descendants) not in (0, 2) for descendants in children.values()):
        raise ValueError("SharpTNI input requires a binary tree")

    postorder: list[int] = []
    seen: set[int] = set()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(node_ids) + 100))

    def visit(node: int) -> None:
        if node in seen:
            raise ValueError(f"Cycle or repeated child detected at node {node}")
        seen.add(node)
        for child in children[node]:
            visit(child)
        postorder.append(node)

    visit(root)
    if seen != node_ids:
        raise ValueError("The KPTracer edge list is disconnected")

    sharp_ids = {node: index for index, node in enumerate(postorder, start=1)}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hosts.txt").open("w") as file:
        for name in STATE_NAMES:
            file.write(f"{name} 0 1\n")

    with (output_dir / "ptree.txt").open("w") as file:
        for node in postorder:
            descendants = children[node]
            child_ids = [str(sharp_ids[child]) for child in descendants]
            if not child_ids:
                child_ids = ["0", "0"]
                label = leaf_labels[node]
            else:
                label = ancestral_labels[node]
            file.write(
                f"{normalized_times[node]:.12g} "
                f"{child_ids[0]} {child_ids[1]} {label + 1}\n"
            )

    (output_dir / "assumptions.txt").write_text(
        "SharpTNI exploratory KPTracer run\n"
        "Hosts: inferred location states T1, L1, L2, L3, and S1.\n"
        "Entry/removal times: [0,1] for all states; no per-location "
        "epidemiological intervals are available in these inputs.\n"
        "Tree: normalized nodes_0.csv times reversed and scaled to [0,1], "
        "with tips at time 1.\n"
        "Leaf labels: 1461.labeling; internal labels: "
        "0-T-0.location.labeling.\n"
        "Interpretation: exploratory state-transition analysis, not a "
        "validated host-to-host infection model.\n"
    )
    return len(node_ids), len(leaves)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the normalized KPTracer tree to SharpTNI inputs."
    )
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=repo_root / "data" / "KPTracer-Data" / "processed_data",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    node_count, leaf_count = prepare_inputs(args.processed_dir, args.output_dir)
    print(
        f"Prepared SharpTNI inputs for {node_count} nodes and {leaf_count} tips "
        f"in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
