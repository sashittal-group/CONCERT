from __future__ import annotations

import argparse
from importlib import import_module
import sys
from collections.abc import Sequence


ANALYSES = {
    "tls-leaf-counts": "src.analyses.tls.leaf_counts",
    "profile-preprocess": "src.analyses.common.profile_preprocess",
    "profile-plot": "src.analyses.common.profile_plot",
    "simulate-coordinate-tree": "src.analyses.simulation.simulate_coordinate_tree",
    "generate-linear-transition-settings": (
        "src.analyses.simulation.generate_linear_transition_settings"
    ),
    "generate-star-transition-settings": (
        "src.analyses.simulation.generate_star_transition_settings"
    ),
    "generate-coordinate-settings": "src.analyses.simulation.generate_coord_settings",
    "run-mach2-simulated-tree": "src.analyses.simulation.run_mach2_simulated_tree",
    "prepare-beam-barcodes": "src.analyses.simulation.prepare_beam_barcodes",
    "run-beam-fixed-tree": "src.analyses.simulation.run_beam_fixed_tree",
    "combine-beam-summaries": "src.analyses.simulation.combine_beam_summaries",
    "joint-state-specific-transition-times": (
        "src.analyses.crc11.joint_state_specific_transition_times"
    ),
    "plot-linear-transition-inference-accuracy": (
        "src.analyses.simulation.plot_linear_transition_inference_accuracy"
    ),
    "extract-star-transition-times": (
        "src.analyses.simulation.extract_star_transition_times"
    ),
    "population-transition-time-search": (
        "src.analyses.simulation.population_transition_time_search"
    ),
    "crc11-upgma": "src.analyses.crc11.preprocessing.upgma",
}

INFERENCE_MODES = {
    "--given-transition-times": "src.concert.infer_parameters",
    "--given-edge-specific-transition-times": (
        "src.concert.infer_given_transitions"
    ),
    "--continue": "src.concert.infer_parameters",
    "--progressive-schedule": "src.concert.progressive_transition_schedule",
    "--state-specific-transitions": (
        "src.concert.infer_state_specific_transitions"
    ),
}


def _add_selectors(parser: argparse.ArgumentParser) -> None:
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--given-transition-times",
        action="store_true",
        help="fit parameters with fixed transition times",
    )
    modes.add_argument(
        "--given-edge-specific-transition-times",
        action="store_true",
        help="fit parameters using observed edge-specific transitions",
    )
    modes.add_argument(
        "--continue",
        dest="continue_inference",
        action="store_true",
        help="continue inference from saved parameter estimates",
    )
    modes.add_argument(
        "--progressive-schedule",
        action="store_true",
        help="estimate transition times with progressive grid sweeps",
    )
    modes.add_argument(
        "--state-specific-transitions",
        action="store_true",
        help="fit the state-specific transition model",
    )
    modes.add_argument(
        "--analysis",
        choices=ANALYSES,
        help="run a named analysis task",
    )


def _get_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.concert",
        description=(
            "Run CONCERT inference or a repository analysis. With no selector, "
            "jointly infer transition times and model parameters."
        ),
        epilog=(
            "Inference selectors are mutually exclusive. Use "
            "`python -m src.concert --given-transition-times --help` for "
            "mode-specific options."
        ),
    )
    _add_selectors(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        _get_dispatch_parser().print_help()
        return

    parser = argparse.ArgumentParser(
        prog="python -m src.concert",
        description="Select one CONCERT inference mode or analysis task.",
        add_help=False,
    )
    _add_selectors(parser)
    selection, forwarded = parser.parse_known_args(arguments)

    if selection.analysis is not None:
        module_name = ANALYSES[selection.analysis]
        continue_inference = False
    elif selection.given_transition_times:
        module_name = INFERENCE_MODES["--given-transition-times"]
        continue_inference = False
    elif selection.given_edge_specific_transition_times:
        module_name = INFERENCE_MODES["--given-edge-specific-transition-times"]
        continue_inference = False
    elif selection.continue_inference:
        module_name = INFERENCE_MODES["--continue"]
        continue_inference = True
    elif selection.progressive_schedule:
        module_name = INFERENCE_MODES["--progressive-schedule"]
        continue_inference = False
    elif selection.state_specific_transitions:
        module_name = INFERENCE_MODES["--state-specific-transitions"]
        continue_inference = False
    else:
        module_name = "src.concert.infer_joint_transition_times"
        continue_inference = False

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *forwarded]
        application = import_module(module_name)
        if continue_inference:
            application.main(require_initial_parameters=True)
        else:
            application.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
