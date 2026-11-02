from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from src.results.cell_state_colors import CELL_STATE_COLORS
from src.results.tls.config import load_config
from src.results.tls.process_state_coloring import OUTPUT_FILENAME


SEGMENT_COLUMNS = (
    "subtree",
    "segment_id",
    "segment_parent_node",
    "segment_child_node",
    "segment_start_time",
    "segment_end_time",
    "level",
    "state",
    "state_probability",
    "most_likely_state",
    "most_likely_state_probability",
)
LEAF_MARKER_SIZE = 16


def _tree_layout(
    graph: nx.DiGraph,
) -> tuple[dict[str, float], dict[str, float], list[str], float]:
    roots = [node for node, degree in graph.in_degree() if degree == 0]
    if len(roots) != 1 or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Each colored subtree must be a rooted directed tree.")
    root = roots[0]
    heights = {root: 0.0}

    def assign_heights(node: str) -> None:
        for child in graph.successors(node):
            heights[child] = (
                heights[node] + float(graph.edges[node, child]["length"])
            )
            assign_heights(child)

    assign_heights(root)
    if len(heights) != graph.number_of_nodes():
        raise ValueError("The subtree contains nodes disconnected from its root.")

    x_positions: dict[str, float] = {}
    leaf_order = [
        node
        for node in nx.dfs_postorder_nodes(graph, source=root)
        if graph.out_degree(node) == 0 and node != root
    ]
    x_positions.update(
        {leaf: float(index) for index, leaf in enumerate(leaf_order)}
    )

    def assign_x(node: str) -> float:
        if node in x_positions:
            return x_positions[node]
        children = list(graph.successors(node))
        child_positions = [assign_x(child) for child in children]
        x_positions[node] = float(np.mean(child_positions))
        return x_positions[node]

    assign_x(root)
    if len(x_positions) != graph.number_of_nodes():
        raise ValueError("The subtree contains nodes disconnected from its root.")
    max_height = max(heights.values())
    y_positions = {node: -height for node, height in heights.items()}
    return x_positions, y_positions, leaf_order, max_height


def _draw_subtree(
    axis: Axes,
    segments: pd.DataFrame,
    dataset: str,
    subtree_name: str,
    state_names: tuple[str, ...],
    transition_markers: tuple[tuple[float, str], ...] = (),
    state_colors: dict[str, str] | None = None,
    title: str | None = None,
    font_size: float | None = None,
) -> float:
    if segments.empty:
        raise ValueError(f"No edge segments to plot for {subtree_name}")
    segments = segments.drop_duplicates("segment_id").sort_values(
        ["level", "segment_id"]
    )
    graph = nx.DiGraph()
    for row in segments.itertuples(index=False):
        parent = str(row.segment_parent_node)
        child = str(row.segment_child_node)
        start_time = float(row.segment_start_time)
        end_time = float(row.segment_end_time)
        for node, node_time in ((parent, start_time), (child, end_time)):
            if node in graph:
                if not np.isclose(
                    graph.nodes[node]["time"], node_time, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        f"Inconsistent time for node {node} in {subtree_name}"
                    )
            else:
                graph.add_node(node, time=node_time)
        graph.add_edge(
            parent,
            child,
            length=start_time - end_time,
            **row._asdict(),
        )

    x_positions, y_positions, leaf_order, max_height = _tree_layout(graph)
    state_colors = CELL_STATE_COLORS if state_colors is None else state_colors
    state_counts = {state: 0 for state in state_names}
    probabilities = []
    axis.scatter(
        [x_positions[node] for node in graph],
        [y_positions[node] for node in graph],
        color="black",
        zorder=1,
        s=5,
        alpha=0.3,
    )
    for parent, child, data in graph.edges(data=True):
        state = str(data["most_likely_state"])
        if state not in state_colors:
            raise ValueError(f"Unknown state label {state!r} in {subtree_name}")
        probability = float(data["most_likely_state_probability"])
        state_counts[state] += 1
        probabilities.append(probability)
        axis.plot(
            [x_positions[parent], x_positions[child]],
            [y_positions[parent], y_positions[parent]],
            color=state_colors[state],
            alpha=0.5,
            zorder=2,
        )
        axis.plot(
            [x_positions[child], x_positions[child]],
            [y_positions[parent], y_positions[child]],
            color=state_colors[state],
            alpha=0.5,
            zorder=2,
        )

    leaf_colors = []
    for leaf in leaf_order:
        incoming_edges = list(graph.in_edges(leaf, data=True))
        if len(incoming_edges) != 1:
            raise ValueError(
                f"Expected one incoming edge for leaf {leaf} in {subtree_name}"
            )
        state = str(incoming_edges[0][2]["most_likely_state"])
        if state not in state_colors:
            raise ValueError(f"Unknown state label {state!r} in {subtree_name}")
        leaf_colors.append(state_colors[state])
    axis.scatter(
        [x_positions[leaf] for leaf in leaf_order],
        [y_positions[leaf] for leaf in leaf_order],
        c=leaf_colors,
        s=LEAF_MARKER_SIZE,
        edgecolors="white",
        linewidths=0.35,
        zorder=5,
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=state_colors[state],
            linewidth=3,
            label=state,
        )
        for state in state_names
    ]
    axis.legend(
        handles=legend,
        title="Most likely segment state",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=len(state_names),
        fontsize=font_size,
    )
    axis.set_title(
        title if title is not None else f"{dataset}: {subtree_name}",
        fontsize=10 if font_size is None else font_size,
    )
    axis.set_ylabel("Time", fontsize=font_size)
    major_ticks = np.linspace(0, max_height, 21)
    axis.set_yticks(-major_ticks)
    tick_step = max_height / (len(major_ticks) - 1)
    tick_precision = (
        max(2, int(np.ceil(-np.log10(tick_step)))) if tick_step > 0 else 2
    )
    axis.set_yticklabels(
        [f"{time:.{tick_precision}f}" for time in major_ticks],
        fontsize=8 if font_size is None else font_size,
    )
    axis.set_ylim(-max_height * 1.02, 0.0)
    axis.set_xticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    for transition_time, color in transition_markers:
        if not 0.0 < transition_time < max_height:
            raise ValueError(
                f"Transition time {transition_time} is outside the subtree time range."
            )
        axis.axhline(
            -transition_time,
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            zorder=4,
        )
    return max_height


def _plot_subtree(
    segments: pd.DataFrame,
    output_path: Path,
    dataset: str,
    subtree_name: str,
    state_names: tuple[str, ...],
    dpi: int,
) -> dict[str, object]:
    figure, axis = plt.subplots(figsize=(8, 5))
    _draw_subtree(
        axis,
        segments,
        dataset,
        subtree_name,
        state_names,
    )
    graph = nx.DiGraph()
    for row in segments.itertuples(index=False):
        graph.add_edge(str(row.segment_parent_node), str(row.segment_child_node))
    probabilities = segments["most_likely_state_probability"].astype(float)
    state_counts = segments["most_likely_state"].value_counts()
    figure.subplots_adjust(top=0.76, bottom=0.1, left=0.12, right=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "dataset": dataset,
        "subtree": subtree_name,
        "figure": output_path.as_posix(),
        "segment_count": graph.number_of_edges(),
        "leaf_count": sum(
            graph.out_degree(node) == 0 for node in graph.nodes
        ),
        "mean_most_likely_probability": float(np.mean(probabilities)),
        **{
            f"segments_{state}": int(state_counts.get(state, 0))
            for state in state_names
        },
    }


def plot_dataset(
    dataset: str,
    state_data_path: Path,
    input_dir: Path,
    output_dir: Path,
    state_names: tuple[str, ...],
    dpi: int = 180,
) -> Path:
    segments_by_subtree = pd.read_csv(
        state_data_path,
        usecols=list(SEGMENT_COLUMNS),
    )
    if segments_by_subtree.empty:
        raise ValueError(f"No segment state data in {state_data_path}")

    required_states = set(state_names)
    if set(segments_by_subtree["state"].unique()) != required_states:
        raise ValueError(f"State labels in {state_data_path} do not match {state_names}")

    index_rows = []
    for subtree_name, subtree_segments in segments_by_subtree.groupby(
        "subtree",
        sort=True,
    ):
        subtree_dir = input_dir / Path(subtree_name)
        nodes_path = subtree_dir / "nodes.csv"
        if not nodes_path.is_file():
            raise FileNotFoundError(f"Missing source subtree nodes: {nodes_path}")
        output_path = output_dir / Path(subtree_name).with_suffix(".png")
        row = _plot_subtree(
            subtree_segments,
            output_path,
            dataset,
            str(subtree_name),
            state_names,
            dpi,
        )
        row["figure"] = output_path.relative_to(output_dir).as_posix()
        index_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "plot_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    print(
        f"{dataset}: wrote {len(index_rows)} colored subtree plots to {output_dir}"
    )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot subtree edges colored by their most likely state."
    )
    parser.add_argument("--dataset", choices=("TLS", "TLSC"), required=True)
    parser.add_argument("--state-data", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_fixed_sampling_0.1.yaml"),
    )
    arguments = parser.parse_args()
    dataset_root = Path("data/troupe") / arguments.dataset
    coloring_dir = dataset_root / "fixed_sampling_0.1" / "state_coloring"
    state_data_path = arguments.state_data or (coloring_dir / OUTPUT_FILENAME)
    input_dir = arguments.input_dir or dataset_root / "subtrees_kept_states"
    output_dir = arguments.output_dir or coloring_dir / "plots"
    config = load_config(arguments.config)
    plot_dataset(
        arguments.dataset,
        state_data_path,
        input_dir,
        output_dir,
        config.state_names,
        dpi=arguments.dpi,
    )


if __name__ == "__main__":
    main()
