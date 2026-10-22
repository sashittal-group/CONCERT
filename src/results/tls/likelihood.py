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
) -> float:
    u_result = calculate_Us(
        birth_rates=birth_rates,
        death_rates=death_rates,
        sampling_probabilities=sampling_probabilities,
        transition_probabilities=transition_probabilities,
        transition_times=torch.tensor(context.transition_times, dtype=torch.float64),
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
        if no_stem:
            root_log_likelihood = crown_conditioned_root_log_likelihood(
                likelihood_by_root_state[0],
                u_result.root_U[tree_index],
            )
        else:
            root_log_likelihood = likelihood_by_root_state[0, 0]
        total += float(root_log_likelihood)
    return total
