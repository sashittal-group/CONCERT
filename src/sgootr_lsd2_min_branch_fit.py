from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from Bio import Phylo

if __package__:
    from .plotters import color_map, draw_ultrametric_tree
    from .sgootr_unpooled_fit import fit, load_tree
else:
    from plotters import color_map, draw_ultrametric_tree
    from sgootr_unpooled_fit import fit, load_tree


ROOT = Path(__file__).resolve().parents[1]
INPUT_TREE = ROOT / "data/sgootr/CRC11.nwk"
MINIMUM_BRANCH_LENGTH = 0.01
STEM_LENGTH = 0.1
SEQUENCE_LENGTH = 1000
ROUNDING_FACTOR = 100
OUTPUT_TREE = ROOT / "data/sgootr/CRC11_ultrametric_min_branch_0p01.nwk"
OUTPUT_FIGURE = ROOT / "data/sgootr/CRC11_ultrametric_min_branch_0p01.png"
OUTPUT_DIR = ROOT / "data/sgootr/inference/min_branch_0p01"
OUTPUT_JSON = ROOT / "data/sgootr/inference/inference_unpooled_one_state_min_branch_0p01.json"
color_map.update({"NC": "gray", "PT": "blue", "LN": "orange"})


def run_lsd2(output_dir: Path) -> Path:
    executable = shutil.which("lsd2")
    if executable is None:
        raise FileNotFoundError("LSD2 was not found on PATH.")

    tree_with_stem = Phylo.read(INPUT_TREE, "newick")
    if tree_with_stem.root.branch_length not in (None, 0.0):
        raise ValueError(
            f"Input tree already has a stem length: {tree_with_stem.root.branch_length}"
        )
    tree_with_stem.root.branch_length = STEM_LENGTH

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        stem_tree_path = temporary_path / "CRC11_with_stem.nwk"
        Phylo.write(
            tree_with_stem,
            stem_tree_path,
            "newick",
            format_branch_length="%1.10f",
        )
        serialized_stem = Phylo.read(stem_tree_path, "newick").root.branch_length
        if serialized_stem is None or not math.isclose(
            serialized_stem, STEM_LENGTH, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Failed to preserve the stem length in {stem_tree_path}.")

        output_prefix = temporary_path / "CRC11_min_branch_0p01"
        command = [
            executable,
            "-i",
            str(stem_tree_path.resolve()),
            "-o",
            str(output_prefix),
            "-s",
            str(SEQUENCE_LENGTH),
            "-r",
            "a",
            "-l",
            "-1",
            "-u",
            str(MINIMUM_BRANCH_LENGTH),
            "-U",
            str(MINIMUM_BRANCH_LENGTH),
            "-R",
            str(ROUNDING_FACTOR),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path = output_dir / "lsd2.log"
        log_path.write_text(
            f"$ {' '.join(command)}\n\n"
            f"Exit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LSD2 failed with exit code {result.returncode}; see {log_path}."
            )

        dated_tree_path = Path(f"{output_prefix}.date.nexus")
        if not dated_tree_path.is_file():
            raise FileNotFoundError(
                f"LSD2 did not create the expected tree: {dated_tree_path}"
            )

        dated_tree = Phylo.read(dated_tree_path, "nexus")

    source_tips = {tip.name for tip in Phylo.read(INPUT_TREE, "newick").get_terminals()}
    dated_tips = {tip.name for tip in dated_tree.get_terminals()}
    if source_tips != dated_tips:
        raise ValueError("The LSD2 output tip labels do not match the input tree.")

    OUTPUT_TREE.parent.mkdir(parents=True, exist_ok=True)
    Phylo.write(dated_tree, OUTPUT_TREE, "newick", format_branch_length="%1.10f")
    return OUTPUT_TREE


def export_inference_csvs(tree_path: Path, output_dir: Path) -> tuple[Path, Path, dict[str, float]]:
    tree = Phylo.read(tree_path, "newick")
    clades = list(tree.find_clades(order="preorder"))
    node_ids = {id(clade): index for index, clade in enumerate(clades)}
    stem_node_id = len(clades)

    depths = tree.depths()
    tip_depths = [depths[tip] for tip in tree.get_terminals()]
    maximum_tip_depth = max(tip_depths)
    tip_depth_range = max(tip_depths) - min(tip_depths)
    if tip_depth_range > 1e-5:
        raise ValueError(
            f"LSD2 output is not ultrametric: root-to-tip range is {tip_depth_range:.3g}."
        )

    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    edge_lengths: list[tuple[float, bool]] = []
    with nodes_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "T", "is_leaf", "state_id"])
        for clade in clades:
            depth = depths[clade] + STEM_LENGTH
            time = maximum_tip_depth + STEM_LENGTH - depth
            writer.writerow(
                [
                    node_ids[id(clade)],
                    f"{time:.15g}",
                    int(clade.is_terminal()),
                    -1,
                ]
            )
        writer.writerow(
            [stem_node_id, f"{maximum_tip_depth + STEM_LENGTH:.15g}", 0, -1]
        )

    with edges_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent", "child", "branch_length"])
        for parent in clades:
            for child in parent.clades:
                length = child.branch_length
                if length is None or not math.isfinite(length) or length < 0.0:
                    raise ValueError(f"Invalid branch length on child {child.name!r}.")
                writer.writerow([node_ids[id(parent)], node_ids[id(child)], f"{length:.15g}"])
                edge_lengths.append((length, child.is_terminal()))
        writer.writerow([stem_node_id, node_ids[id(tree.root)], f"{STEM_LENGTH:.15g}"])
        edge_lengths.append((STEM_LENGTH, False))

    minimum_internal = min(length for length, terminal in edge_lengths if not terminal)
    minimum_external = min(length for length, terminal in edge_lengths if terminal)
    minimum_observed = min(length for length, _ in edge_lengths)
    if minimum_observed < MINIMUM_BRANCH_LENGTH - 1e-8:
        raise ValueError(
            f"LSD2 produced a branch shorter than {MINIMUM_BRANCH_LENGTH}: "
            f"{minimum_observed:.15g}"
        )

    return nodes_path, edges_path, {
        "root_to_tip_range": tip_depth_range,
        "minimum_observed_branch_length": minimum_observed,
        "minimum_internal_branch_length": minimum_internal,
        "minimum_external_branch_length": minimum_external,
    }


def render_adjusted_figure(tree_path: Path) -> Path:
    tree = Phylo.read(tree_path, "newick")
    clades = list(tree.find_clades(order="preorder"))
    node_ids = {id(clade): index for index, clade in enumerate(clades)}
    depths = tree.depths()
    maximum_tip_depth = max(depths[tip] for tip in tree.get_terminals())
    prefix_to_state = {"gray": "NC", "blue": "PT", "orange": "LN"}

    tip_states: dict[int, str] = {}
    for tip in tree.get_terminals():
        state = next(
            (
                state
                for prefix, state in prefix_to_state.items()
                if tip.name is not None and tip.name.startswith(f"{prefix}_")
            ),
            None,
        )
        if state is None:
            raise ValueError(f"No state mapping for tip label: {tip.name}")
        tip_states[id(tip)] = state

    clade_states: dict[int, str | None] = {}
    for clade in reversed(clades):
        if clade.is_terminal():
            clade_states[id(clade)] = tip_states[id(clade)]
        else:
            child_states = {clade_states[id(child)] for child in clade.clades}
            clade_states[id(clade)] = (
                child_states.pop() if len(child_states) == 1 else None
            )

    graph = nx.DiGraph()
    for clade in clades:
        node = node_ids[id(clade)]
        graph.add_node(
            node,
            label=clade.name or "",
            terminal=clade.is_terminal(),
            state=clade_states[id(clade)],
            T=maximum_tip_depth - depths[clade],
        )
        for child in clade.clades:
            if child.branch_length is None:
                raise ValueError(f"Missing branch length below node {node}.")
            graph.add_edge(
                node,
                node_ids[id(child)],
                length=child.branch_length,
            )

    stem_node_id = len(clades)
    graph.add_node(
        stem_node_id,
        label="stem",
        terminal=False,
        state="NC",
        T=maximum_tip_depth + STEM_LENGTH,
    )
    graph.add_edge(stem_node_id, node_ids[id(tree.root)], length=STEM_LENGTH)

    node_colors = [
        (node, data["state"])
        for node, data in graph.nodes(data=True)
        if data["state"] is not None
    ]
    draw_ultrametric_tree(
        graph,
        root=stem_node_id,
        add_labels=False,
        figsize=(20, 6),
        node_colors=node_colors,
    )
    figure = plt.gcf()
    axis = figure.axes[0]
    axis.set_title("SGOOTR tree dated with LSD2 minimum branch length 0.01")
    figure.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT_FIGURE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-date SGOOTR with an LSD2 0.01 branch floor and refit one state."
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dated_tree_path = run_lsd2(OUTPUT_DIR)
    figure_path = render_adjusted_figure(dated_tree_path)
    nodes_path, edges_path, branch_diagnostics = export_inference_csvs(
        dated_tree_path, OUTPUT_DIR
    )
    tree = load_tree(nodes_path, edges_path)
    fit_result = fit(tree, max_iter=args.max_iter)

    result: dict[str, object] = {
        "method": "whole tree treated as one state; no transitions",
        "root_normalized": True,
        "input_tree": str(INPUT_TREE.relative_to(ROOT)),
        "dated_tree": str(OUTPUT_TREE.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
        "nodes_csv": str(nodes_path.relative_to(ROOT)),
        "edges_csv": str(edges_path.relative_to(ROOT)),
        "lsd2_options": {
            "sequence_length": SEQUENCE_LENGTH,
            "rooting_method": "a",
            "input_branch_collapse_threshold": -1,
            "minimum_internal_branch_length": MINIMUM_BRANCH_LENGTH,
            "minimum_external_branch_length": MINIMUM_BRANCH_LENGTH,
            "rounding_factor": ROUNDING_FACTOR,
            "stem_length": STEM_LENGTH,
        },
        "root_node": tree.root,
        "root_time": tree.root_time,
        "num_nodes": tree.num_nodes,
        "num_edges": len(tree.edges),
        "birth_events": tree.births,
        "sampled_leaves": tree.leaves,
        "zero_length_edges": tree.zero_length_edges,
        "multifurcating_nodes": tree.multifurcating_nodes,
        "branch_diagnostics": branch_diagnostics,
        "polytomy_assumption": (
            "Each node with k children contributes k-1 binary birth events."
        ),
        **fit_result,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    summary_path = OUTPUT_JSON.with_suffix(".out")
    summary_path.write_text(
        "\n".join(
            [
                f"LL={result['log_likelihood']:.15f}",
                f"birth: {result['birth_rate']:.15g}",
                f"death: {result['death_rate']:.15g}",
                f"sampling: {result['sampling_probability']:.15g}",
                f"root: {tree.root} at time {tree.root_time:.15g}",
                f"birth_events: {tree.births}",
                f"sampled_leaves: {tree.leaves}",
                f"zero_length_edges: {tree.zero_length_edges}",
                f"minimum_branch_length: "
                f"{branch_diagnostics['minimum_observed_branch_length']:.15g}",
            ]
        )
        + "\n"
    )
    print(summary_path.read_text(), end="")
    print(f"Adjusted tree: {dated_tree_path}")
    print(f"Adjusted figure: {figure_path}")
    print(f"JSON results: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
