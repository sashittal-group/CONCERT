from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.data import DatasetContext, load_dataset_context
from src.results.tls.likelihood import evaluate_total_log_likelihood
from src.results.profile_tools import ParameterProfile, parameter_grid, profile_rows


def build_parameter_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        growth = float(context.birth_rates[index] - context.death_rates[index])
        upper_growth = config.max_birth_rate - float(context.death_rates[index])
        profiles.append(
            ParameterProfile(
                f"growth[{state_name}]",
                "growth",
                index,
                growth,
                parameter_grid(
                    growth,
                    config.rate_epsilon,
                    upper_growth,
                    config.profile_grid_points,
                    f"growth[{state_name}]",
                ),
            )
        )

    profiles.extend(_birth_rate_profiles(context, config))
    profiles.extend(_death_rate_profiles(context, config))
    profiles.extend(_sampling_rate_profiles(context, config))
    profiles.extend(_scaled_sampling_profiles(context, config))
    profiles.extend(_transition_profiles(context, config))
    profiles.append(_transition_ratio_profile(context, config))
    return profiles


def _birth_rate_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        birth = float(context.birth_rates[index])
        death = float(context.death_rates[index])
        name = f"birth[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "birth",
                index,
                birth,
                parameter_grid(
                    birth,
                    death + config.rate_epsilon,
                    config.max_birth_rate,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _death_rate_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        death = float(context.death_rates[index])
        upper = float(context.birth_rates[index]) - config.rate_epsilon
        lower = min(config.rate_epsilon, death / 10.0)
        name = f"death[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "death",
                index,
                death,
                parameter_grid(
                    death,
                    lower,
                    upper,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _sampling_rate_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    sampling_values = [
        float(value) for value in context.sampling_probabilities
    ]
    shared_rate = all(
        np.isclose(value, sampling_values[0], rtol=1e-8, atol=1e-10)
        for value in sampling_values[1:]
    )
    if shared_rate:
        return [
            ParameterProfile(
                "sampling[shared]",
                "sampling",
                0,
                sampling_values[0],
                parameter_grid(
                    sampling_values[0],
                    config.probability_epsilon,
                    1.0 - config.probability_epsilon,
                    config.profile_grid_points,
                    "sampling[shared]",
                ),
            )
        ]

    profiles = []
    for index, state_name in enumerate(config.state_names):
        name = f"sampling[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "sampling",
                index,
                sampling_values[index],
                parameter_grid(
                    sampling_values[index],
                    config.probability_epsilon,
                    1.0 - config.probability_epsilon,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _scaled_sampling_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        sampling = float(context.sampling_probabilities[index])
        if not 0.0 < sampling < 1.0:
            raise ValueError(f"Sampling probability for {state_name} must be in (0, 1)")
        original = sampling * float(context.birth_rates[index])
        lower = sampling * (
            float(context.death_rates[index]) + config.rate_epsilon
        )
        upper = sampling * config.max_birth_rate
        profiles.append(
            ParameterProfile(
                f"scaled_sampling[{state_name}]",
                "scaled_sampling",
                index,
                original,
                parameter_grid(
                    original,
                    lower,
                    upper,
                    config.profile_grid_points,
                    f"scaled_sampling[{state_name}]",
                ),
            )
        )
    return profiles


def _transition_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for target in range(1, len(config.state_names)):
        original = float(context.transition_probabilities[0, target])
        other_probability = sum(
            float(context.transition_probabilities[0, index])
            for index in range(1, len(config.state_names))
            if index != target
        )
        upper = 1.0 - other_probability - config.probability_epsilon
        name = f"transition[{config.state_names[0]}->{config.state_names[target]}]"
        profiles.append(
            ParameterProfile(
                name,
                "transition",
                target,
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
    return profiles


def _transition_ratio_profile(
    context: DatasetContext,
    config: PipelineConfig,
) -> ParameterProfile:
    probabilities = context.transition_probabilities[0]
    first_probability = float(probabilities[1])
    second_probability = float(probabilities[2])
    epsilon = config.probability_epsilon
    if first_probability <= epsilon or second_probability <= epsilon:
        raise ValueError("NMP outgoing transition probabilities must exceed epsilon")
    outgoing = first_probability + second_probability
    if outgoing <= 2.0 * epsilon:
        raise ValueError("NMP outgoing transition probability is too small to profile")

    lower = epsilon / (outgoing - epsilon)
    upper = 1.0 / lower
    original = first_probability / second_probability
    values = np.unique(
        np.concatenate(
            (
                np.geomspace(lower, upper, config.profile_grid_points),
                np.geomspace(
                    max(lower, original / 5.0),
                    min(upper, original * 5.0),
                    config.profile_grid_points,
                ),
                [original],
            )
        )
    )
    return ParameterProfile(
        "transition_ratio[Somite/NeuralTube]",
        "transition_ratio",
        0,
        original,
        values,
    )


def _candidate_parameters(
    context: DatasetContext,
    profile: ParameterProfile,
    candidate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    birth = context.birth_rates.clone()
    death = context.death_rates.clone()
    sampling = context.sampling_probabilities.clone()
    transitions = context.transition_probabilities.clone()

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
    elif profile.parameter_type == "transition_ratio":
        outgoing = context.transition_probabilities[0, 1] + context.transition_probabilities[0, 2]
        transitions[0, 1] = outgoing * candidate / (1.0 + candidate)
        transitions[0, 2] = outgoing / (1.0 + candidate)
        transitions[0, 0] = 1.0 - transitions[0, 1:].sum()
    elif profile.parameter_type == "transition":
        transitions[0, profile.index] = candidate
        transitions[0, 0] = 1.0 - transitions[0, 1:].sum()
    else:
        raise ValueError(f"Unsupported profile type: {profile.parameter_type}")
    return birth, death, sampling, transitions


def _profile_likelihoods(
    context: DatasetContext,
    profile: ParameterProfile,
    *,
    normalize_root_likelihood: bool = False,
) -> np.ndarray:
    values = []
    for candidate in profile.values:
        parameters = _candidate_parameters(context, profile, float(candidate))
        values.append(
            evaluate_total_log_likelihood(
                context,
                *parameters,
                normalize_root_likelihood=normalize_root_likelihood,
            )
        )
    return np.asarray(values, dtype=float)


def preprocess_dataset(dataset: str, config: PipelineConfig) -> None:
    context = load_dataset_context(
        config.dataset_root(dataset),
        config.state_names,
        config.dataset_inference_path(dataset),
    )
    baseline = evaluate_total_log_likelihood(
        context,
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
    )
    profile_data_rows = []
    range_rows = []
    metadata_rows = []
    for profile in build_parameter_profiles(context, config):
        log_likelihoods = _profile_likelihoods(context, profile)
        rows, ranges = profile_rows(
            dataset,
            profile,
            log_likelihoods,
            baseline,
            config.likelihood_fractions,
        )
        profile_data_rows.extend(rows)
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
    pd.DataFrame(profile_data_rows).to_csv(
        output_dir / config.profile_filename,
        index=False,
    )
    pd.DataFrame(range_rows).to_csv(output_dir / config.ranges_filename, index=False)
    pd.DataFrame(metadata_rows).to_csv(
        output_dir / config.metadata_filename, index=False
    )
    print(f"{dataset}: baseline log likelihood {baseline:.6f}; saved profiles to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess TLS/TLSC composite-parameter sensitivity profiles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    preprocess_dataset(args.dataset, load_config(args.config))


if __name__ == "__main__":
    main()
