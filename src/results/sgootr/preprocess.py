from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.results.profile_tools import ParameterProfile, parameter_grid, profile_rows
from src.results.sgootr.data import load_dataset_context
from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.data import DatasetContext
from src.results.tls.likelihood import evaluate_total_log_likelihood


def _growth_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        original = float(context.birth_rates[index] - context.death_rates[index])
        upper = config.max_birth_rate - float(context.death_rates[index])
        name = f"growth[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "growth",
                index,
                original,
                parameter_grid(
                    original,
                    config.rate_epsilon,
                    upper,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _birth_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        original = float(context.birth_rates[index])
        lower = float(context.death_rates[index]) + config.rate_epsilon
        name = f"birth[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "birth",
                index,
                original,
                parameter_grid(
                    original,
                    lower,
                    config.max_birth_rate,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _death_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        original = float(context.death_rates[index])
        upper = float(context.birth_rates[index]) - config.rate_epsilon
        name = f"death[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "death",
                index,
                original,
                parameter_grid(
                    original,
                    config.rate_epsilon,
                    upper,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _sampling_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    values = [float(value) for value in context.sampling_probabilities]
    if all(
        np.isclose(value, values[0], rtol=1e-8, atol=1e-10)
        for value in values[1:]
    ):
        name = "sampling[shared]"
        return [
            ParameterProfile(
                name,
                "sampling",
                0,
                values[0],
                parameter_grid(
                    values[0],
                    config.probability_epsilon,
                    1.0 - config.probability_epsilon,
                    config.profile_grid_points,
                    name,
                ),
            )
        ]

    return [
        ParameterProfile(
            f"sampling[{state_name}]",
            "sampling",
            index,
            values[index],
            parameter_grid(
                values[index],
                config.probability_epsilon,
                1.0 - config.probability_epsilon,
                config.profile_grid_points,
                f"sampling[{state_name}]",
            ),
        )
        for index, state_name in enumerate(config.state_names)
    ]


def _scaled_sampling_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        sampling = float(context.sampling_probabilities[index])
        original = sampling * float(context.birth_rates[index])
        lower = sampling * (
            float(context.death_rates[index]) + config.rate_epsilon
        )
        upper = sampling * config.max_birth_rate
        name = f"scaled_sampling[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "scaled_sampling",
                index,
                original,
                parameter_grid(
                    original,
                    lower,
                    upper,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _active_transition_edges(context: DatasetContext) -> list[tuple[int, int]]:
    num_states = len(context.birth_rates)
    return [
        (source, target)
        for source in range(num_states)
        for target in range(num_states)
        if source != target
        and float(context.transition_probabilities[source, target]) > 0.0
    ]


def _transition_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    num_states = len(context.birth_rates)
    for source, target in _active_transition_edges(context):
        original = float(context.transition_probabilities[source, target])
        other_outgoing = sum(
            float(context.transition_probabilities[source, index])
            for index in range(num_states)
            if index not in {source, target}
        )
        upper = 1.0 - other_outgoing - config.probability_epsilon
        name = (
            f"transition[{config.state_names[source]}->"
            f"{config.state_names[target]}]"
        )
        profiles.append(
            ParameterProfile(
                name,
                "transition",
                source * num_states + target,
                original,
                parameter_grid(
                    original,
                    config.probability_epsilon,
                    upper,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    if not profiles:
        raise ValueError("No non-diagonal SGOOTR transitions were inferred.")
    return profiles


def build_parameter_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    return [
        *_growth_profiles(context, config),
        *_birth_profiles(context, config),
        *_death_profiles(context, config),
        *_sampling_profiles(context, config),
        *_scaled_sampling_profiles(context, config),
        *_transition_profiles(context, config),
    ]


def _candidate_parameters(
    context: DatasetContext,
    profile: ParameterProfile,
    candidate: float,
    config: PipelineConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    birth = context.birth_rates.clone()
    death = context.death_rates.clone()
    sampling = context.sampling_probabilities.clone()
    transitions = context.transition_probabilities.clone()
    num_states = len(config.state_names)

    if profile.parameter_type == "growth":
        birth[profile.index] = death[profile.index] + candidate
    elif profile.parameter_type == "birth":
        birth[profile.index] = candidate
    elif profile.parameter_type == "death":
        death[profile.index] = candidate
    elif profile.parameter_type == "sampling":
        if profile.name == "sampling[shared]":
            sampling[:] = candidate
        else:
            sampling[profile.index] = candidate
    elif profile.parameter_type == "scaled_sampling":
        birth[profile.index] = candidate / sampling[profile.index]
    elif profile.parameter_type == "transition":
        source, target = divmod(profile.index, num_states)
        transitions[source, target] = candidate
        transitions[source, source] = 1.0 - sum(
            transitions[source, index]
            for index in range(num_states)
            if index != source
        )
    else:
        raise ValueError(f"Unsupported profile type: {profile.parameter_type}")
    return birth, death, sampling, transitions


def _profile_likelihoods(
    context: DatasetContext,
    profile: ParameterProfile,
    config: PipelineConfig,
) -> np.ndarray:
    values = []
    for candidate in profile.values:
        parameters = _candidate_parameters(
            context,
            profile,
            float(candidate),
            config,
        )
        values.append(
            evaluate_total_log_likelihood(
                context,
                *parameters,
                no_stem=True,
            )
        )
    return np.asarray(values, dtype=float)


def preprocess_dataset(dataset: str, config: PipelineConfig) -> None:
    context = load_dataset_context(config.dataset_root(dataset), config.state_names)
    baseline = evaluate_total_log_likelihood(
        context,
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
        no_stem=True,
    )
    profile_rows_by_parameter = []
    range_rows = []
    metadata_rows = []
    for profile in build_parameter_profiles(context, config):
        likelihoods = _profile_likelihoods(context, profile, config)
        rows, ranges = profile_rows(
            dataset,
            profile,
            likelihoods,
            baseline,
            config.likelihood_fractions,
        )
        profile_rows_by_parameter.extend(rows)
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

    output_dir = config.dataset_output_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profile_rows_by_parameter).to_csv(
        output_dir / config.profile_filename,
        index=False,
    )
    pd.DataFrame(range_rows).to_csv(output_dir / config.ranges_filename, index=False)
    pd.DataFrame(metadata_rows).to_csv(
        output_dir / config.metadata_filename,
        index=False,
    )
    print(
        f"{dataset}: baseline crown log likelihood {baseline:.6f}; "
        f"saved profiles to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SGOOTR crown-conditioned parameter profiles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    preprocess_dataset(
        args.dataset,
        load_config(args.config, section="sgootr_profiles"),
    )


if __name__ == "__main__":
    main()
