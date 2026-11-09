from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.analyses.crc11.preprocessing.birth_shift import shift_birth_events
from src.analyses.crc11.preprocessing.dating import date_tree_with_lsd2
from src.analyses.crc11.preprocessing.transition_anchor import (
    anchor_birth_events_to_transitions,
)
from src.analyses.crc11.preprocessing.transitions import (
    infer_transition_events,
    project_transition_events,
    retain_transition_times,
)
from src.analyses.crc11.preprocessing.tree import (
    TreeRepresentation,
    build_tree_representation,
    export_stage,
    load_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_TREE = PROJECT_ROOT / "data" / "sgootr" / "CRC11.nwk"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "sgootr" / "processed"


def run_pipeline(
    input_tree_path: Path,
    output_root: Path,
    *,
    lsd2_executable: str | None = None,
    mach2_executable: str | None = None,
    sequence_length: int = 1000,
    dating_minimum_branch_length: float = 0.001,
    birth_minimum_branch_length: float = 0.025,
    post_transition_terminal_floor: float = 0.025,
    rounding_factor: int = 100,
    ultrametric_tolerance: float = 1e-5,
) -> dict[str, dict[str, Path]]:
    raw_tree = load_tree(input_tree_path)
    raw_stage = build_tree_representation(raw_tree)
    outputs = {
        "input": export_stage(
            raw_stage,
            output_root / "input",
            title="Input SGOOTR tree",
        )
    }

    ultrametric_tree_path = output_root / "ultrametric" / "tree.nwk"
    ultrametric_tree = date_tree_with_lsd2(
        input_tree_path,
        ultrametric_tree_path,
        PROJECT_ROOT,
        executable=lsd2_executable,
        sequence_length=sequence_length,
        minimum_branch_length=dating_minimum_branch_length,
        rounding_factor=rounding_factor,
        ultrametric_tolerance=ultrametric_tolerance,
    )
    ultrametric_stage = build_tree_representation(ultrametric_tree)
    _validate_ultrametric(ultrametric_stage, ultrametric_tolerance)
    outputs["ultrametric"] = export_stage(
        ultrametric_stage,
        output_root / "ultrametric",
        title="LSD2 ultrametric tree",
    )

    transition_events = infer_transition_events(
        ultrametric_stage,
        output_root / "ultrametric",
        PROJECT_ROOT,
        executable=mach2_executable,
    )
    outputs["ultrametric"] = export_stage(
        ultrametric_stage,
        output_root / "ultrametric",
        transition_events,
        title="Ultrametric tree with inferred transitions",
    )

    birth_shifted_tree = shift_birth_events(
        ultrametric_tree,
        birth_minimum_branch_length,
        ultrametric_tolerance,
    )
    birth_shifted_stage = build_tree_representation(birth_shifted_tree)
    shifted_events = project_transition_events(
        ultrametric_stage,
        birth_shifted_stage,
        transition_events,
    )
    outputs["birth_shifted"] = export_stage(
        birth_shifted_stage,
        output_root / "birth_shifted_min_branch",
        shifted_events,
        title="Birth-shifted tree",
    )

    transition_anchored_tree, scale_factor = anchor_birth_events_to_transitions(
        ultrametric_stage,
        [event.transition_time for event in transition_events],
        post_transition_terminal_floor,
    )
    transition_anchored_stage = build_tree_representation(transition_anchored_tree)
    anchored_events = retain_transition_times(
        transition_anchored_stage,
        transition_events,
    )
    outputs["transition_anchored"] = export_stage(
        transition_anchored_stage,
        output_root / "transition_anchored",
        anchored_events,
        title="Transition-anchored tree",
    )

    print(f"Processed {input_tree_path} into {len(outputs)} tree stages.")
    print(
        f"Inferred {len(transition_events)} transition events at "
        f"{len({event.transition_time for event in transition_events})} unique times."
    )
    print(f"Transition-anchored post-transition scale factor: {scale_factor:.10g}.")
    for stage_name, paths in outputs.items():
        print(f"{stage_name}: {paths['nodes']}, {paths['edges']}, {paths['plot']}")
    return outputs


def _validate_ultrametric(
    stage: TreeRepresentation,
    tolerance: float,
) -> None:
    tip_depths = [
        stage.depths[node]
        for node, data in stage.graph.nodes(data=True)
        if data["terminal"]
    ]
    tip_depth_range = max(tip_depths) - min(tip_depths)
    if not math.isfinite(tip_depth_range) or tip_depth_range > tolerance:
        raise ValueError(
            f"Tree is not ultrametric: root-to-tip range {tip_depth_range:.3g} "
            f"exceeds {tolerance:g}."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process each SGOOTR tree stage and export CSVs and plots."
    )
    parser.add_argument("--input-tree", type=Path, default=DEFAULT_INPUT_TREE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lsd2-executable")
    parser.add_argument("--mach2-executable")
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument(
        "--dating-minimum-branch-length",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--birth-minimum-branch-length",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--post-transition-terminal-floor",
        type=float,
        default=0.025,
    )
    parser.add_argument("--rounding-factor", type=int, default=100)
    parser.add_argument("--ultrametric-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run_pipeline(
        arguments.input_tree,
        arguments.output_root,
        lsd2_executable=arguments.lsd2_executable,
        mach2_executable=arguments.mach2_executable,
        sequence_length=arguments.sequence_length,
        dating_minimum_branch_length=arguments.dating_minimum_branch_length,
        birth_minimum_branch_length=arguments.birth_minimum_branch_length,
        post_transition_terminal_floor=arguments.post_transition_terminal_floor,
        rounding_factor=arguments.rounding_factor,
        ultrametric_tolerance=arguments.ultrametric_tolerance,
    )


if __name__ == "__main__":
    main()
