from __future__ import annotations

import torch

from src.closed_form_given_transitions import calculate_Us, calculate_log_likelihood
from src.mle2_given_transitions import crown_conditioned_root_log_likelihood
from src.results.tls.data import DatasetContext


def evaluate_total_log_likelihood(
    context: DatasetContext,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    transition_probabilities: torch.Tensor,
    no_stem: bool = False,
    normalize_root_likelihood: bool = False,
) -> float:
    if no_stem and normalize_root_likelihood:
        raise ValueError(
            "Crown conditioning cannot be combined with root-likelihood normalization"
        )
    transition_times = torch.tensor(context.transition_times, dtype=torch.float64)
    if sampling_probabilities.ndim == 2 or transition_probabilities.ndim == 3:
        if sampling_probabilities.ndim == 2:
            expected_shape = (
                len(context.sensitivity_inputs),
                len(birth_rates),
            )
            if tuple(sampling_probabilities.shape) != expected_shape:
                raise ValueError(
                    "Per-tree sampling probabilities must have shape "
                    f"{expected_shape}; found {tuple(sampling_probabilities.shape)}"
                )
        if transition_probabilities.ndim == 3:
            expected_shape = (
                len(context.sensitivity_inputs),
                len(birth_rates),
                len(birth_rates),
            )
            if tuple(transition_probabilities.shape) != expected_shape:
                raise ValueError(
                    "Per-tree transition probabilities must have shape "
                    f"{expected_shape}; found {tuple(transition_probabilities.shape)}"
                )
        if (
            transition_probabilities.ndim == 3
            and (
                context.tree_transition_times is None
                or len(context.tree_transition_times) != len(context.sensitivity_inputs)
            )
        ):
            raise ValueError(
                "Per-tree transition matrices require transition times for each tree"
            )
        total = 0.0
        for tree_index, subtree in enumerate(context.sensitivity_inputs):
            tree_sampling = (
                sampling_probabilities[tree_index]
                if sampling_probabilities.ndim == 2
                else sampling_probabilities
            )
            tree_transitions = (
                transition_probabilities[tree_index]
                if transition_probabilities.ndim == 3
                else transition_probabilities
            )
            tree_times = (
                context.tree_transition_times[tree_index]
                if transition_probabilities.ndim == 3
                else subtree.transition_times
            )
            transition_tensor = torch.tensor(tree_times, dtype=torch.float64)
            u_result = calculate_Us(
                birth_rates=birth_rates,
                death_rates=death_rates,
                sampling_probabilities=tree_sampling,
                transition_probabilities=tree_transitions,
                transition_times=transition_tensor,
                root_times=[subtree.root_time],
            )
            likelihood_by_root_state = calculate_log_likelihood(
                datastructure=subtree.datastructure,
                birth_rates=birth_rates,
                death_rates=death_rates,
                transition_probabilities=tree_transitions,
                sampling_probabilities=tree_sampling,
                Cs=u_result.Cs,
                transition_times=tree_times,
            )
            root_log_likelihood = likelihood_by_root_state[0, 0]
            if no_stem:
                root_log_likelihood = crown_conditioned_root_log_likelihood(
                    likelihood_by_root_state[0],
                    u_result.root_U[0],
                )
            elif normalize_root_likelihood:
                root_log_likelihood -= torch.log1p(-u_result.root_U[0, 0])
            else:
                root_log_likelihood = likelihood_by_root_state[0, 0]
            total += float(root_log_likelihood)
        return total

    u_result = calculate_Us(
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_probabilities=sampling_probabilities,
        transition_probabilities=transition_probabilities,
        transition_times=transition_times,
        root_times=context.root_times,
    )
    total = 0.0
    for tree_index, subtree in enumerate(context.sensitivity_inputs):
        likelihood_by_root_state = calculate_log_likelihood(
            datastructure=subtree.datastructure,
            birth_rates=birth_rates,
            death_rates=death_rates,
            transition_probabilities=transition_probabilities,
            sampling_probabilities=sampling_probabilities,
            Cs=u_result.Cs[:len(subtree.transition_times) + 1],
            transition_times=subtree.transition_times,
        )
        root_log_likelihood = likelihood_by_root_state[0, 0]
        if no_stem:
            root_log_likelihood = crown_conditioned_root_log_likelihood(
                likelihood_by_root_state[0],
                u_result.root_U[tree_index],
            )
        elif normalize_root_likelihood:
            root_log_likelihood -= torch.log1p(-u_result.root_U[tree_index, 0])
        else:
            root_log_likelihood = likelihood_by_root_state[0, 0]
        total += float(root_log_likelihood)
    return total
