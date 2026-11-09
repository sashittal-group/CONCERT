from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analyses.common.profile_tools import ParameterProfile, parameter_grid, profile_rows
from src.analyses.kptracer.data import load_dataset_context
from src.analyses.common.profile_config import PipelineConfig, load_config
from src.analyses.common.profile_data import DatasetContext
from src.analyses.common.profile_likelihood import evaluate_total_log_likelihood


def _growth_limit(
    context: DatasetContext,
    index: int,
    config: PipelineConfig,
) -> float:
    growth = float(context.birth_rates[index] - context.death_rates[index])
    if not np.isfinite(growth) or growth <= 0.0:
        raise ValueError(
            f"KPTracer requires positive net growth for state "
            f"{config.state_names[index]} to build rate profiles"
        )
    return max(
        growth * config.rate_profile_growth_span,
        _effective_rate_epsilon(context, index, config) * 100.0,
    )


def _effective_rate_epsilon(
    context: DatasetContext,
    index: int,
    config: PipelineConfig,
) -> float:
    rate_scale = max(
        abs(float(context.birth_rates[index])),
        abs(float(context.death_rates[index])),
        1.0,
    )
    return max(config.rate_epsilon, float(np.spacing(rate_scale)) * 8.0)


def _growth_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    for index, state_name in enumerate(config.state_names):
        original = float(context.birth_rates[index] - context.death_rates[index])
        name = f"growth[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "growth",
                index,
                original,
                parameter_grid(
                    original,
                    _effective_rate_epsilon(context, index, config),
                    _growth_limit(context, index, config),
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
        birth = float(context.birth_rates[index])
        death = float(context.death_rates[index])
        lower = death + _effective_rate_epsilon(context, index, config)
        upper = min(
            config.max_birth_rate,
            death + _growth_limit(context, index, config),
        )
        name = f"birth[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "birth",
                index,
                birth,
                parameter_grid(
                    birth,
                    lower,
                    upper,
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
        birth = float(context.birth_rates[index])
        death = float(context.death_rates[index])
        name = f"death[{state_name}]"
        profiles.append(
            ParameterProfile(
                name,
                "death",
                index,
                death,
                parameter_grid(
                    death,
                    max(
                        _effective_rate_epsilon(context, index, config),
                        birth - _growth_limit(context, index, config),
                    ),
                    birth - _effective_rate_epsilon(context, index, config),
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
    if all(np.isclose(value, values[0], rtol=1e-8, atol=1e-10) for value in values[1:]):
        return [
            ParameterProfile(
                "sampling[shared]",
                "sampling",
                0,
                values[0],
                parameter_grid(
                    values[0],
                    config.probability_epsilon,
                    1.0 - config.probability_epsilon,
                    config.profile_grid_points,
                    "sampling[shared]",
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
        birth = float(context.birth_rates[index])
        death = float(context.death_rates[index])
        name = f"scaled_sampling[{state_name}]"
        upper_birth = min(
            config.max_birth_rate,
            death + _growth_limit(context, index, config),
        )
        profiles.append(
            ParameterProfile(
                name,
                "scaled_sampling",
                index,
                sampling * birth,
                parameter_grid(
                    sampling * birth,
                    sampling
                    * (death + _effective_rate_epsilon(context, index, config)),
                    sampling * upper_birth,
                    config.profile_grid_points,
                    name,
                ),
            )
        )
    return profiles


def _transition_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    num_states = len(config.state_names)
    for source in range(num_states):
        for target in range(num_states):
            if source == target:
                continue
            original = float(context.transition_probabilities[source, target])
            if original <= 0.0:
                continue
            other_outgoing = sum(
                float(context.transition_probabilities[source, index])
                for index in range(num_states)
                if index not in {source, target}
            )
            upper = 1.0 - other_outgoing - config.probability_epsilon
            name = f"transition[{config.state_names[source]}->{config.state_names[target]}]"
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
        raise ValueError("No non-diagonal KPTracer transitions were inferred")
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


def preprocess_dataset(dataset: str, config: PipelineConfig) -> None:
    context = load_dataset_context(config.data_root, config.state_names)
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
        likelihoods = []
        for candidate in profile.values:
            parameters = _candidate_parameters(
                context,
                profile,
                float(candidate),
                config,
            )
            likelihoods.append(
                evaluate_total_log_likelihood(
                    context,
                    *parameters,
                    no_stem=True,
                )
            )
        rows, ranges = profile_rows(
            dataset,
            profile,
            np.asarray(likelihoods, dtype=float),
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
        description="Preprocess KPTracer crown-conditioned parameter profiles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument("--dataset", default="KPTracer")
    args = parser.parse_args()
    preprocess_dataset(
        args.dataset,
        load_config(args.config, section="kptracer_profiles"),
    )


if __name__ == "__main__":
    main()
