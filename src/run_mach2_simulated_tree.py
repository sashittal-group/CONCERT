from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
import subprocess

from src.process_data.sgootr.executables import resolve_executable


def _read_tree_inputs(
    nodes_path: Path,
    edges_path: Path,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    node_rows = {}
    with nodes_path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not {"node_id", "is_leaf", "state_id"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{nodes_path} must contain node_id, is_leaf, and state_id")
        for row in reader:
            node_id = int(row["node_id"])
            if node_id in node_rows:
                raise ValueError(f"Duplicate node {node_id} in {nodes_path}")
            leaf_value = row["is_leaf"].strip().lower()
            if leaf_value not in {"0", "1", "false", "true"}:
                raise ValueError(f"Invalid is_leaf value for node {node_id} in {nodes_path}")
            is_leaf = leaf_value in {"1", "true"}
            state_id = int(row["state_id"])
            if is_leaf and not 0 <= state_id <= 3:
                raise ValueError(
                    f"Leaf node {node_id} has invalid state_id {state_id} in {nodes_path}"
                )
            node_rows[node_id] = (is_leaf, state_id)

    if not node_rows:
        raise ValueError(f"No nodes found in {nodes_path}")
    mach2_id_by_node = {
        node_id: mach2_id
        for mach2_id, node_id in enumerate(node_rows, start=1)
    }

    edges = []
    children = set()
    parent_by_child = {}
    with edges_path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not {"parent", "child"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{edges_path} must contain parent and child columns")
        for row in reader:
            parent = int(row["parent"])
            child = int(row["child"])
            if parent not in node_rows or child not in node_rows or parent == child:
                raise ValueError(f"Invalid edge {parent}->{child} in {edges_path}")
            if child in parent_by_child:
                raise ValueError(f"Node {child} has multiple parents in {edges_path}")
            parent_by_child[child] = parent
            children.add(child)
            edges.append((parent, child))

    roots = set(node_rows).difference(children)
    if len(roots) != 1 or len(edges) != len(node_rows) - 1:
        raise ValueError(f"Expected one rooted tree in {edges_path}")
    root = roots.pop()
    adjacency = {node_id: [] for node_id in node_rows}
    for parent, child in edges:
        adjacency[parent].append(child)
    reachable = set()
    pending = [root]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(adjacency[node_id])
    if reachable != set(node_rows):
        raise ValueError(f"Tree in {edges_path} is not connected")

    leaves = {
        node_id for node_id, (is_leaf, _) in node_rows.items() if is_leaf
    }
    if not leaves or leaves != set(node_rows).difference(
        parent for parent, child in edges
    ):
        raise ValueError(f"Leaf labels do not match terminal nodes in {nodes_path}")

    mach2_edges = [
        (mach2_id_by_node[parent], mach2_id_by_node[child])
        for parent, child in edges
    ]
    mach2_labels = [
        (mach2_id_by_node[node_id], state_id)
        for node_id, (is_leaf, state_id) in node_rows.items()
        if is_leaf
    ]
    return mach2_edges, mach2_labels


def run_mach2_simulated_tree(
    nodes_path: Path,
    edges_path: Path,
    output_dir: Path,
    mach2_executable: str | None = None,
    threads: int = 1,
    timelimit: int | None = None,
) -> Path:
    if threads < 1:
        raise ValueError("MACH2 threads must be at least 1")
    if timelimit is not None and timelimit < 1:
        raise ValueError("MACH2 timelimit must be at least 1 second")
    project_root = Path(__file__).resolve().parents[1]
    mach2 = resolve_executable("mach2", mach2_executable, project_root)
    edges, labels = _read_tree_inputs(nodes_path, edges_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = output_dir / "tree.tsv"
    labeling_path = output_dir / "labeling.tsv"
    mach2_output = output_dir / "mach2_output"
    if mach2_output.exists() and any(mach2_output.iterdir()):
        raise FileExistsError(f"MACH2 output directory is not empty: {mach2_output}")
    mach2_output.mkdir(exist_ok=True)
    tree_path.write_text("".join(f"{parent}\t{child}\n" for parent, child in edges))
    labeling_path.write_text("".join(f"{node}\t{state}\n" for node, state in labels))

    command = [
        str(mach2),
        str(tree_path),
        str(labeling_path),
        "-t",
        str(threads),
    ]
    if timelimit is not None:
        command.extend(("--timelimit", str(timelimit)))
    command.extend(("-o", str(mach2_output)))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path = output_dir / "mach2.log"
    log_path.write_text(
        f"$ {shlex.join(command)}\n\n"
        f"Exit code: {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MACH2 failed with exit code {result.returncode}; see {log_path}"
        )

    refined_trees = sorted(mach2_output.glob("*-T-0.refined.tree"))
    if len(refined_trees) != 1:
        raise FileNotFoundError(
            f"Expected one first-solution MACH2 tree in {mach2_output}; "
            f"found {len(refined_trees)}"
        )
    prefix = Path(str(refined_trees[0])[: -len(".refined.tree")])
    missing = [
        path
        for path in (
            Path(f"{prefix}.nodes"),
            Path(f"{prefix}.location.labeling"),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"MACH2 output is incomplete: {missing}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MACH2 on one simulated tree with sampled-tip state labels."
    )
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mach2-executable")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timelimit", type=int)
    args = parser.parse_args()
    run_mach2_simulated_tree(
        args.nodes,
        args.edges,
        args.output_dir,
        mach2_executable=args.mach2_executable,
        threads=args.threads,
        timelimit=args.timelimit,
    )


if __name__ == "__main__":
    main()
