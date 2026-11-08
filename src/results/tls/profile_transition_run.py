from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.results.profile_tools import ParameterProfile, profile_rows
from src.results.sgootr.profile_combined_inference import (
    load_combined_dataset_context,
)
from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.likelihood import evaluate_total_log_likelihood
from src.results.tls.plot import plot_profile_data
from src.results.tls.preprocess import (
    _profile_likelihoods,
    build_parameter_profiles,
)


TRANSITION_RUN = re.compile(r"inference_transition_times_(\d+)")
LOG_LIKELIHOOD = re.compile(r"(?:ITER\s+\d+\s+)?LL=(\S+)")


def _read_log_likelihood(path: Path) -> float:
    matches = [
        match.group(1)
        for line in path.read_text().splitlines()
        if (match := LOG_LIKELIHOOD.fullmatch(line))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one log likelihood in {path}; found {len(matches)}")
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(f"Invalid log likelihood in {path}") from error
    if not math.isfinite(value):
        raise ValueError(f"Log likelihood in {path} must be finite")
    return value


def _fixed_sampling_profiles(
    profiles: list[ParameterProfile],
    sampling_probabilities: np.ndarray,
) -> list[ParameterProfile]:
    if not np.allclose(sampling_probabilities, 0.1, rtol=0.0, atol=1e-12):
        raise ValueError("This profiling workflow requires fixed sampling at 0.1.")
    sampling_profiles = [
        profile for profile in profiles if profile.parameter_type == "sampling"
    ]
    if len(sampling_profiles) != 1 or sampling_profiles[0].name != "sampling[shared]":
        raise ValueError("Expected one shared fixed-sampling parameter.")

    return [
        replace(profile, values=np.asarray([profile.original_value], dtype=float))
        if profile.parameter_type == "sampling"
        else profile
        for profile in profiles
    ]


def _profile_config(
    config: PipelineConfig,
    dataset: str,
    dataset_label: str,
    output_dir: Path,
) -> PipelineConfig:
    plotting = dict(config.plotting)
    plotting.update(
        {
            "output_basename": f"{dataset}_composite_sensitivity",
            "parameter_output_basenames": {
                "birth": f"{dataset}_birth_rates",
                "death": f"{dataset}_death_rates",
                "transition": f"{dataset}_transition_probabilities",
                "sampling": f"{dataset}_sampling_rates",
            },
            "dataset_colors": {
                dataset: config.plotting["dataset_colors"][dataset_label]
            },
            "dataset_labels": {dataset: f"{dataset_label} (100 transition times)"},
        }
    )
    return replace(
        config,
        datasets=(dataset,),
        data_root=output_dir / "data",
        preprocessing_root=output_dir / "data",
        figure_root=output_dir / "figures",
        plotting=plotting,
    )


def profile_transition_run(
    inference_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> list[Path]:
    match = TRANSITION_RUN.search(inference_path.name)
    if match is None:
        raise ValueError(
            "Expected an inference filename containing "
            "'inference_transition_times_<count>'."
        )
    transition_count = int(match.group(1))
    if transition_count != 100:
        raise ValueError("This profile workflow is scoped to 100 transition times.")

    dataset_label = next(
        (
            parent.name
            for parent in inference_path.parents
            if parent.name in {"TLS", "TLSC"}
        ),
        None,
    )
    if dataset_label is None:
        raise ValueError(
            f"Could not determine whether {inference_path} belongs to TLS or TLSC."
        )
    dataset = f"{dataset_label}_transition_times_100"
    context = load_combined_dataset_context(
        input_dir,
        inference_path,
        config.state_names,
        use_per_edge_transition_data=False,
        allow_stem_root=True,
    )
    sampling = context.sampling_probabilities.detach().cpu().numpy()
    profiles = _fixed_sampling_profiles(
        build_parameter_profiles(context, config),
        sampling,
    )
    baseline = evaluate_total_log_likelihood(
        context,
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
        normalize_root_likelihood=True,
    )
    fitted_log_likelihood = _read_log_likelihood(inference_path)
    if not np.isclose(baseline, fitted_log_likelihood, rtol=1e-8, atol=1e-8):
        raise ValueError(
            "Profile baseline does not match the inference result: "
            f"{baseline:.15g} != {fitted_log_likelihood:.15g}."
        )

    profile_data_rows = []
    range_rows = []
    metadata_rows = []
    for profile in profiles:
        likelihoods = _profile_likelihoods(
            context,
            profile,
            normalize_root_likelihood=True,
        )
        profile_data, ranges = profile_rows(
            dataset,
            profile,
            likelihoods,
            baseline,
            config.likelihood_fractions,
        )
        profile_data_rows.extend(profile_data)
        range_rows.extend(ranges)
        metadata_rows.append(
            {
                "dataset": dataset,
                "parameter": profile.name,
                "parameter_type": profile.parameter_type,
                "parameter_index": profile.index,
                "original_value": profile.original_value,
            }
        )

    plot_config = _profile_config(config, dataset, dataset_label, output_dir)
    data_dir = plot_config.dataset_output_dir(dataset)
    data_dir.mkdir(parents=True, exist_ok=True)
    profiles_frame = pd.DataFrame(profile_data_rows)
    ranges_frame = pd.DataFrame(range_rows)
    metadata_frame = pd.DataFrame(metadata_rows)
    profiles_frame.to_csv(data_dir / config.profile_filename, index=False)
    ranges_frame.to_csv(data_dir / config.ranges_filename, index=False)
    metadata_frame.to_csv(data_dir / config.metadata_filename, index=False)

    figures = plot_profile_data(
        plot_config,
        {dataset: ranges_frame},
        {dataset: metadata_frame},
    )
    print(
        f"Baseline log likelihood: {baseline:.15g}; "
        f"saved profile tables under {data_dir}."
    )
    for figure in figures:
        print(f"Wrote {figure}")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile TLS/TLSC parameters for a fixed-sampling 100-transition-time fit."
        )
    )
    parser.add_argument(
        "inference",
        type=Path,
        help="The TLS or TLSC 100-transition-time inference snapshot.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing the dataset's nodes_# and edges_# files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Profile output directory; defaults to profiles/ beside the inference.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_fixed_sampling_0.1.yaml"),
    )
    arguments = parser.parse_args()
    input_dir = (
        arguments.input_dir
        or arguments.inference.parent.parent / "subtrees_kept_states"
    )
    dataset = next(
        (
            parent.name
            for parent in arguments.inference.parents
            if parent.name in {"TLS", "TLSC"}
        ),
        None,
    )
    if dataset is None:
        raise ValueError(
            f"Could not determine whether {arguments.inference} belongs to TLS or TLSC."
        )
    output_dir = arguments.output_dir or (
        Path("outputs/tls/fixed_sampling_0.1/transition_profiles")
        / dataset
    )
    profile_transition_run(
        arguments.inference,
        input_dir,
        output_dir,
        load_config(arguments.config),
    )


if __name__ == "__main__":
    main()
