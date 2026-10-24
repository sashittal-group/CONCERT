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


def _profile_max_birth_rate(
    context: DatasetContext,
    config: PipelineConfig,
) -> float:
    highest_birth_rate = float(context.birth_rates.max())
    return max(config.max_birth_rate, highest_birth_rate * 1.1)


def _growth_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    max_birth_rate = _profile_max_birth_rate(context, config)
    for index, state_name in enumerate(config.state_names):
        original = float(context.birth_rates[index] - context.death_rates[index])
        upper = max_birth_rate - float(context.death_rates[index])
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
    max_birth_rate = _profile_max_birth_rate(context, config)
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
                    max_birth_rate,
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
    if context.sampling_probabilities.ndim == 2:
        profiles = []
        state_specific = context.tree_sampling_scope == "per_tree_by_state"
        for tree_index, tree_values in enumerate(context.sampling_probabilities):
            shared_across_states = torch.allclose(
                tree_values,
                tree_values[0].expand_as(tree_values),
                rtol=1e-10,
                atol=1e-12,
            )
            if state_specific or not shared_across_states:
                for state_index, state_name in enumerate(config.state_names):
                    original = float(tree_values[state_index])
                    name = f"sampling[tree_{tree_index},{state_name}]"
                    profiles.append(
                        ParameterProfile(
                            name,
                            "sampling",
                            tree_index * len(config.state_names) + state_index,
                            original,
                            parameter_grid(
                                original,
                                config.probability_epsilon,
                                1.0 - config.probability_epsilon,
                                config.profile_grid_points,
                                name,
                            ),
                        )
                    )
            else:
                original = float(tree_values[0])
                name = f"sampling[tree_{tree_index}]"
                profiles.append(
                    ParameterProfile(
                        name,
                        "sampling",
                        tree_index,
                        original,
                        parameter_grid(
                            original,
                            config.probability_epsilon,
                            1.0 - config.probability_epsilon,
                            config.profile_grid_points,
                            name,
                        ),
                    )
                )
        return profiles

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


def _sampling_ratio_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    sampling = context.sampling_probabilities
    if sampling.ndim == 2:
        sampling_by_tree = sampling
    elif sampling.ndim == 1:
        sampling_by_tree = sampling.unsqueeze(0)
    else:
        raise ValueError("Sampling probabilities must be a vector or tree matrix")

    reference_index = config.state_names.index("NC")
    epsilon = config.probability_epsilon
    profiles = []
    for tree_index, tree_sampling in enumerate(sampling_by_tree):
        reference_probability = float(tree_sampling[reference_index])
        for target_name in ("PT", "LN"):
            target_index = config.state_names.index(target_name)
            target_probability = float(tree_sampling[target_index])
            pair_total = reference_probability + target_probability
            minimum_reference = max(epsilon, pair_total - (1.0 - epsilon))
            maximum_reference = min(1.0 - epsilon, pair_total - epsilon)
            if minimum_reference >= maximum_reference:
                raise ValueError(
                    f"Sampling probabilities for {target_name}/NC have no feasible "
                    f"ratio range in tree {tree_index}"
                )
            lower = pair_total / maximum_reference - 1.0
            upper = pair_total / minimum_reference - 1.0
            original = target_probability / reference_probability
            name = f"sampling_ratio[{target_name}/NC]"
            profile_index = target_index
            if sampling.ndim == 2:
                name = f"sampling_ratio[tree_{tree_index},{target_name}/NC]"
                profile_index = tree_index * len(config.state_names) + target_index
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
            profiles.append(
                ParameterProfile(
                    name,
                    "sampling_ratio",
                    profile_index,
                    original,
                    values,
                )
            )
    return profiles


def _birth_ratio_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    birth_rates = context.birth_rates
    reference_index = config.state_names.index("NC")
    reference_rate = float(birth_rates[reference_index])
    max_birth_rate = _profile_max_birth_rate(context, config)
    profiles = []
    for target_name in ("PT", "LN"):
        target_index = config.state_names.index(target_name)
        target_rate = float(birth_rates[target_index])
        pair_total = reference_rate + target_rate
        minimum_reference = max(
            float(context.death_rates[reference_index]) + config.rate_epsilon,
            pair_total - max_birth_rate,
        )
        maximum_reference = min(
            max_birth_rate,
            pair_total
            - float(context.death_rates[target_index])
            - config.rate_epsilon,
        )
        if minimum_reference >= maximum_reference:
            raise ValueError(
                f"Birth rates for {target_name}/NC have no feasible ratio range"
            )
        lower = pair_total / maximum_reference - 1.0
        upper = pair_total / minimum_reference - 1.0
        original = target_rate / reference_rate
        name = f"birth_ratio[{target_name}/NC]"
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
        profiles.append(
            ParameterProfile(
                name,
                "birth_ratio",
                target_index,
                original,
                values,
            )
        )
    return profiles


def _scaled_sampling_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    max_birth_rate = _profile_max_birth_rate(context, config)
    sampling_probabilities = context.sampling_probabilities
    if sampling_probabilities.ndim == 2:
        sampling_probabilities = sampling_probabilities.mean(dim=0)
    for index, state_name in enumerate(config.state_names):
        sampling = float(sampling_probabilities[index])
        original = sampling * float(context.birth_rates[index])
        lower = sampling * (
            float(context.death_rates[index]) + config.rate_epsilon
        )
        upper = sampling * max_birth_rate
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


def _active_transition_edges(
    context: DatasetContext,
    tree_index: int | None = None,
) -> list[tuple[int, int]]:
    num_states = len(context.birth_rates)
    transition_probabilities = context.transition_probabilities
    if transition_probabilities.ndim == 3:
        transition_probabilities = (
            transition_probabilities[tree_index]
            if tree_index is not None
            else transition_probabilities.mean(dim=0)
        )
    return [
        (source, target)
        for source in range(num_states)
        for target in range(num_states)
        if source != target
        and float(transition_probabilities[source, target]) > 0.0
    ]


def _transition_profiles(
    context: DatasetContext,
    config: PipelineConfig,
) -> list[ParameterProfile]:
    profiles = []
    num_states = len(context.birth_rates)
    tree_specific = context.transition_probabilities.ndim == 3
    matrices = (
        context.transition_probabilities
        if tree_specific
        else context.transition_probabilities.unsqueeze(0)
    )
    for tree_index, matrix in enumerate(matrices):
        for source, target in _active_transition_edges(
            context,
            tree_index if tree_specific else None,
        ):
            original = float(matrix[source, target])
            other_outgoing = sum(
                float(matrix[source, index])
                for index in range(num_states)
                if index not in {source, target}
            )
            upper = 1.0 - other_outgoing - config.probability_epsilon
            edge_name = (
                f"{config.state_names[source]}->{config.state_names[target]}"
            )
            name = (
                f"transition[tree_{tree_index},{edge_name}]"
                if tree_specific
                else f"transition[{edge_name}]"
            )
            profiles.append(
                ParameterProfile(
                    name,
                    "transition",
                    (
                        tree_index * num_states * num_states
                        + source * num_states
                        + target
                        if tree_specific
                        else source * num_states + target
                    ),
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
        elif sampling.ndim == 2:
            if context.tree_sampling_scope == "per_tree_by_state":
                tree_index, state_index = divmod(profile.index, num_states)
                sampling[tree_index, state_index] = candidate
            elif "," in profile.name:
                tree_index, state_index = divmod(profile.index, num_states)
                sampling[tree_index, state_index] = candidate
            else:
                sampling[profile.index, :] = candidate
        else:
            sampling[profile.index] = candidate
    elif profile.parameter_type == "sampling_ratio":
        reference_index = config.state_names.index("NC")
        if sampling.ndim == 2:
            tree_index, target_index = divmod(profile.index, num_states)
            reference_probability = float(sampling[tree_index, reference_index])
            target_probability = float(sampling[tree_index, target_index])
            pair_total = reference_probability + target_probability
            sampling[tree_index, reference_index] = pair_total / (1.0 + candidate)
            sampling[tree_index, target_index] = (
                pair_total * candidate / (1.0 + candidate)
            )
        else:
            reference_probability = float(sampling[reference_index])
            target_probability = float(sampling[profile.index])
            pair_total = reference_probability + target_probability
            sampling[reference_index] = pair_total / (1.0 + candidate)
            sampling[profile.index] = pair_total * candidate / (1.0 + candidate)
    elif profile.parameter_type == "birth_ratio":
        reference_index = config.state_names.index("NC")
        pair_total = birth[reference_index] + birth[profile.index]
        birth[reference_index] = pair_total / (1.0 + candidate)
        birth[profile.index] = pair_total * candidate / (1.0 + candidate)
    elif profile.parameter_type == "scaled_sampling":
        state_sampling = (
            sampling[:, profile.index].mean()
            if sampling.ndim == 2
            else sampling[profile.index]
        )
        birth[profile.index] = candidate / state_sampling
    elif profile.parameter_type == "transition":
        if transitions.ndim == 3:
            tree_index, edge_index = divmod(
                profile.index,
                num_states * num_states,
            )
            source, target = divmod(edge_index, num_states)
            transition_matrix = transitions[tree_index]
        else:
            source, target = divmod(profile.index, num_states)
            transition_matrix = transitions
        transition_matrix[source, target] = candidate
        transition_matrix[source, source] = 1.0 - sum(
            transition_matrix[source, index]
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
    *,
    no_stem: bool = True,
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
                no_stem=no_stem,
            )
        )
    return np.asarray(values, dtype=float)


def preprocess_dataset(dataset: str, config: PipelineConfig) -> None:
    context = load_dataset_context(
        config.dataset_root(dataset),
        config.state_names,
        (
            config.dataset_inference_path(dataset)
            if config.inference_root is not None
            else None
        ),
    )
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
