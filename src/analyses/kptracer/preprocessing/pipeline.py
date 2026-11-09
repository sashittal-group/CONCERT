from __future__ import annotations

import argparse
from pathlib import Path

from src.analyses.kptracer.preprocessing.transitions import process_transition_events
from src.analyses.kptracer.preprocessing.tree import (
    apply_location_states,
    build_tree_graph,
    export_processed_tree,
    normalize_tree,
    write_initial_tree_exports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TREE = PROJECT_ROOT / "data" / "laml" / "problin_1461.nwk"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "laml" / "KPTracer_meta.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "KPTracer-Data" / "processed_data"
DEFAULT_REFINED_TREE = DEFAULT_OUTPUT_DIR / "0-T-0.refined.tree"
DEFAULT_LOCATION_LABELS = DEFAULT_OUTPUT_DIR / "0-T-0.location.labeling"


def run_pipeline(
    newick_path: Path = DEFAULT_TREE,
    metadata_path: Path = DEFAULT_METADATA,
    refined_tree_path: Path = DEFAULT_REFINED_TREE,
    location_labels_path: Path = DEFAULT_LOCATION_LABELS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    minimum_edge_length: float = 0.001,
    target_tip_depth: float = 1.0,
) -> dict[str, Path]:
    for input_path in (
        newick_path,
        metadata_path,
        refined_tree_path,
        location_labels_path,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required KPTracer input not found: {input_path}")

    graph = build_tree_graph(
        newick_path,
        metadata_path,
    )
    outputs = write_initial_tree_exports(graph, output_dir)

    graph = normalize_tree(
        graph,
        minimum_edge_length=minimum_edge_length,
        target_tip_depth=target_tip_depth,
    )
    apply_location_states(graph, location_labels_path)

    nodes_path = output_dir / "nodes_0.csv"
    edges_path = output_dir / "edges_0.csv"
    export_processed_tree(graph, nodes_path, edges_path)
    outputs.update({"nodes": nodes_path, "edges": edges_path})
    outputs.update(
        process_transition_events(
            graph,
            refined_tree_path,
            output_dir,
            minimum_edge_length=minimum_edge_length,
        )
    )

    print(
        f"Processed KPTracer tree into {len(graph)} nodes and "
        f"{graph.number_of_edges()} edges under {output_dir}"
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KPTracer tree and transition preprocessing outputs."
    )
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--refined-tree", type=Path, default=DEFAULT_REFINED_TREE)
    parser.add_argument(
        "--location-labels",
        type=Path,
        default=DEFAULT_LOCATION_LABELS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-edge-length", type=float, default=0.001)
    parser.add_argument("--target-tip-depth", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run_pipeline(
        arguments.tree,
        arguments.metadata,
        arguments.refined_tree,
        arguments.location_labels,
        arguments.output_dir,
        minimum_edge_length=arguments.minimum_edge_length,
        target_tip_depth=arguments.target_tip_depth,
    )


if __name__ == "__main__":
    main()
