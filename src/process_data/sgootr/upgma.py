from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from Bio.Phylo.BaseTree import Tree
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

from src.process_data.sgootr.transitions import infer_transition_events
from src.process_data.sgootr.tree import (
    TreeRepresentation,
    build_tree_representation,
    export_stage,
    load_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_TREE = (
    PROJECT_ROOT / "data" / "sgootr" / "processed" / "input" / "tree.nwk"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sgootr" / "processed" / "upgma"


def calculate_pairwise_distances(
    tree: Tree,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    tips = tree.get_terminals()
    names = tuple(tip.name or "" for tip in tips)
    if not names or any(not name for name in names):
        raise ValueError("Every terminal node must have a nonempty label.")
    if len(set(names)) != len(names):
        raise ValueError("Terminal labels must be unique to build a distance matrix.")

    matrix = [[0.0] * len(names) for _ in names]
    for row, tip in enumerate(tips):
        for column in range(row):
            distance = float(tree.distance(tip, tips[column]))
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    f"Invalid distance between {names[row]!r} and "
                    f"{names[column]!r}: {distance}"
                )
            matrix[row][column] = distance
            matrix[column][row] = distance

    return names, tuple(tuple(row) for row in matrix)


def build_upgma_tree(
    names: tuple[str, ...],
    matrix: tuple[tuple[float, ...], ...],
) -> Tree:
    if len(names) < 2:
        raise ValueError("UPGMA requires at least two terminal nodes.")
    if len(set(names)) != len(names):
        raise ValueError("Terminal labels must be unique to build a distance matrix.")
    if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
        raise ValueError("The distance matrix must be square and match its labels.")

    lower_triangle = []
    for row_index, row in enumerate(matrix):
        for column_index, distance in enumerate(row):
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    "Distance matrix entries must be finite and nonnegative."
                )
            if not math.isclose(distance, matrix[column_index][row_index]):
                raise ValueError("The distance matrix must be symmetric.")
        lower_triangle.append(list(row[: row_index + 1]))

    result = DistanceTreeConstructor().upgma(
        DistanceMatrix(list(names), lower_triangle)
    )
    result.root.branch_length = None
    return result


def scale_tree_to_height(tree: Tree, target_height: float = 1.0) -> float:
    if not math.isfinite(target_height) or target_height <= 0.0:
        raise ValueError("target_height must be finite and positive.")

    tip_heights = [
        float(tree.distance(tree.root, tip))
        for tip in tree.get_terminals()
    ]
    if not tip_heights or any(
        not math.isfinite(height) or height < 0.0 for height in tip_heights
    ):
        raise ValueError("Tree tip heights must be finite and nonnegative.")
    tree_height = max(tip_heights)
    if tree_height <= 0.0:
        raise ValueError("Cannot scale a tree with zero height.")

    scale_factor = target_height / tree_height
    for clade in tree.find_clades():
        if clade is tree.root:
            continue
        branch_length = clade.branch_length
        if branch_length is None or not math.isfinite(branch_length) or branch_length < 0.0:
            raise ValueError(f"Invalid branch length while scaling: {branch_length}")
        clade.branch_length = float(branch_length) * scale_factor
    return scale_factor


def subtract_minimum_pairwise_distance(
    names: tuple[str, ...],
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], float]:
    if len(names) < 2:
        raise ValueError("At least two terminal nodes are required.")
    if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
        raise ValueError("The distance matrix must be square and match its labels.")

    pairwise_distances = [
        matrix[row][column]
        for row in range(len(names))
        for column in range(row)
    ]
    if any(
        not math.isfinite(distance) or distance < 0.0
        for distance in pairwise_distances
    ):
        raise ValueError("Pairwise distances must be finite and nonnegative.")
    minimum_distance = min(pairwise_distances)
    adjusted_matrix = tuple(
        tuple(
            0.0
            if row == column
            else matrix[row][column] - minimum_distance
            for column in range(len(names))
        )
        for row in range(len(names))
    )
    return adjusted_matrix, minimum_distance


def write_distance_matrix(
    names: tuple[str, ...],
    matrix: tuple[tuple[float, ...], ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tip", *names))
        for name, row in zip(names, matrix):
            writer.writerow((name, *(f"{distance:.15g}" for distance in row)))


def run_upgma_preprocessor(
    input_tree_path: Path,
    output_dir: Path,
    *,
    mach2_executable: str | None = None,
    ultrametric_tolerance: float = 1e-5,
    subtract_minimum_distance: bool = False,
    normalize_height: bool = False,
) -> dict[str, Path]:
    input_tree = load_tree(input_tree_path)
    build_tree_representation(input_tree)

    names, matrix = calculate_pairwise_distances(input_tree)
    minimum_distance = None
    if subtract_minimum_distance:
        matrix, minimum_distance = subtract_minimum_pairwise_distance(names, matrix)
    upgma_tree = build_upgma_tree(names, matrix)
    stage = build_tree_representation(upgma_tree)
    original_height = stage.root_age
    time_scale = 1.0
    if normalize_height:
        time_scale = scale_tree_to_height(upgma_tree)
        matrix = tuple(
            tuple(distance * time_scale for distance in row)
            for row in matrix
        )
        stage = build_tree_representation(upgma_tree)
    _validate_ultrametric(stage, ultrametric_tolerance)
    tip_names = {clade.name for clade in stage.tree.get_terminals()}
    if tip_names != set(names):
        raise ValueError("UPGMA output terminal labels do not match the input tree.")

    matrix_path = output_dir / "distance_matrix.csv"
    write_distance_matrix(names, matrix, matrix_path)
    transitions = infer_transition_events(
        stage,
        output_dir,
        PROJECT_ROOT,
        executable=mach2_executable,
    )
    outputs = export_stage(
        stage,
        output_dir,
        transitions,
        title="UPGMA ultrametric SGOOTR tree",
        transition_time_scale=time_scale,
    )
    outputs["distance_matrix"] = matrix_path
    tip_depths = [
        stage.depths[node]
        for node, data in stage.graph.nodes(data=True)
        if data["terminal"]
    ]
    tip_depth_range = max(tip_depths) - min(tip_depths)
    print(
        f"Processed {len(names)} tips with UPGMA into {output_dir}; "
        f"root-to-tip range is {tip_depth_range:.3g}."
    )
    if minimum_distance is not None:
        print(f"Subtracted minimum pairwise distance {minimum_distance:.15g}.")
    if normalize_height:
        print(
            f"Scaled tree height from {original_height:.15g} to "
            f"{stage.root_age:.15g} (factor {time_scale:.15g})."
        )
    return outputs


def _validate_ultrametric(stage: TreeRepresentation, tolerance: float) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("ultrametric_tolerance must be finite and nonnegative.")
    tip_depths = [
        stage.depths[node]
        for node, data in stage.graph.nodes(data=True)
        if data["terminal"]
    ]
    tip_depth_range = max(tip_depths) - min(tip_depths)
    if not math.isfinite(tip_depth_range) or tip_depth_range > tolerance:
        raise ValueError(
            f"UPGMA output is not ultrametric: root-to-tip range "
            f"{tip_depth_range:.3g} exceeds {tolerance:g}."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and export an UPGMA ultrametric tree from a Newick tree."
    )
    parser.add_argument("--input-tree", type=Path, default=DEFAULT_INPUT_TREE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mach2-executable")
    parser.add_argument("--ultrametric-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--subtract-minimum-distance",
        action="store_true",
        help="Subtract the smallest off-diagonal distance from all tip pairs.",
    )
    parser.add_argument(
        "--normalize-height",
        action="store_true",
        help="Scale the tree height and corresponding times to 1.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run_upgma_preprocessor(
        arguments.input_tree,
        arguments.output_dir,
        mach2_executable=arguments.mach2_executable,
        ultrametric_tolerance=arguments.ultrametric_tolerance,
        subtract_minimum_distance=arguments.subtract_minimum_distance,
        normalize_height=arguments.normalize_height,
    )


if __name__ == "__main__":
    main()
