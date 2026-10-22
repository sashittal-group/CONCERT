from __future__ import annotations

import torch

from src.closed_form_coord2 import calculate_Us, calculate_log_likelihood
from src.results.tls.data import DatasetContext


def evaluate_total_log_likelihood(
    context: DatasetContext,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    transition_probabilities: torch.Tensor,
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
    for subtree in context.sensitivity_inputs:
        likelihood_by_root_state = calculate_log_likelihood(
            datastructure=subtree.datastructure,
            birth_rates=birth_rates,
            death_rates=death_rates,
            transition_probabilities=transition_probabilities,
            sampling_probabilities=sampling_probabilities,
            Cs=u_result.Cs[:len(subtree.transition_times) + 1],
            transition_times=subtree.transition_times,
        )
        total += float(likelihood_by_root_state[0, 0])
    return total
