# closed_form_coord.py

from __future__ import annotations

from typing import Sequence

import torch

from src.concert.tree_types import (
    Edge,
    Node,
    PreparedTree,
    LikelihoodResult,
    UResult,
)
from src.concert.tree_utils import prepare_tree


# ============================================================
# CONSTANTS
# ============================================================

NEG_INF = -torch.inf


# ============================================================
# HELPERS
# ============================================================


# ============================================================
# CALCULATE U, FLOW_U, C
# ============================================================


def calculate_Us(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    transition_probabilities: torch.Tensor,
    transition_times: torch.Tensor,
    root_times: Sequence[float] | torch.Tensor,
) -> UResult:
    if birth_rates.ndim != 1:
        raise ValueError("birth_rates must have shape (num_states,)")
    if death_rates.shape != birth_rates.shape:
        raise ValueError("death_rates must have the same shape as birth_rates")
    if sampling_probabilities.shape != birth_rates.shape:
        raise ValueError("sampling_probabilities must have shape (num_states,)")

    num_steps = len(transition_times)
    if num_steps == 0:
        raise ValueError("transition_times cannot be empty")

    root_times = torch.as_tensor(
        root_times,
        dtype=birth_rates.dtype,
        device=birth_rates.device,
    )
    if root_times.ndim != 1 or root_times.numel() == 0:
        raise ValueError("root_times must be a non-empty one-dimensional sequence")

    U0 = 1.0 - sampling_probabilities
    C0 = (U0 - death_rates / birth_rates) / (U0 - 1.0)

    # --------------------------------------------------------
    # Store tensors in Python lists.
    #
    # IMPORTANT:
    # We do not modify a tensor after it has entered the
    # autograd graph.
    # --------------------------------------------------------

    U_values = [U0]

    flow_U_values = [torch.zeros_like(U0)]

    C_values = [C0]

    # --------------------------------------------------------
    # Calculate U and C at each transition epoch.
    rates = birth_rates - death_rates
    for n in range(num_steps):
        dt = transition_times[n]
        if n > 0:
            dt = transition_times[n] - transition_times[n - 1]
        
        exponential = torch.exp(-rates * dt)
        denominator = 1.0 - C_values[n] * exponential
        flow_U_n = 1.0 - rates / (birth_rates * denominator)
        U_n = transition_probabilities @ flow_U_n
        C_n = (U_n - death_rates / birth_rates) / (U_n - 1.0)

        flow_U_values.append(flow_U_n)
        U_values.append(U_n)
        C_values.append(C_n)

    # --------------------------------------------------------
    # Stack after all calculations are complete.
    # --------------------------------------------------------

    U = torch.stack(
        U_values,
        dim=0,
    )

    flow_Us = torch.stack(
        flow_U_values,
        dim=0,
    )

    Cs = torch.stack(
        C_values,
        dim=0,
    )

    epoch_indices = (
        (root_times[:, None] >= transition_times[None, :])
        .to(torch.long)
        .sum(dim=1)
    )
    epoch_times = torch.cat((transition_times.new_zeros(1), transition_times))
    root_epoch_times = epoch_times[epoch_indices]
    root_epoch_Cs = Cs[epoch_indices]
    root_exponential = torch.exp(
        -rates[None, :] * (root_times[:, None] - root_epoch_times[:, None])
    )
    root_U = 1.0 - rates[None, :] / (
        birth_rates[None, :] * (1.0 - root_epoch_Cs * root_exponential)
    )

    return UResult(
        U=U,
        flow_U=flow_Us,
        Cs=Cs,
        root_U=root_U,
    )


def calculate_log_likelihood(
    datastructure: list[dict],
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    transition_probabilities: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    Cs: torch.Tensor,
    transition_times: Sequence[float],
) -> LikelihoodResult:
    """Calculate likelihoods by batching independent tree levels."""

    dtype = birth_rates.dtype
    log_birth = torch.log(birth_rates)
    log_sampling = torch.log(sampling_probabilities.clamp_min(torch.finfo(dtype).tiny))
    log_probabilities = torch.log(
        transition_probabilities.clamp_min(torch.finfo(dtype).tiny)
    )
    rates = birth_rates - death_rates

    level_lls = []
    epoch_times = torch.as_tensor(
        [0.0, *transition_times],
        dtype=dtype,
        device=birth_rates.device,
    )
    # if len(epoch_times) != len(datastructure):
    #     raise ValueError(
    #         "Expected one likelihood level per epoch, including the initial epoch."
    #     )

    for i, level in enumerate(reversed(datastructure)):
        num_births = level["num_births"].to(device=birth_rates.device, dtype=dtype)
        sampling_state_counts = level["num_sampling_by_state"].to(
            device=birth_rates.device,
            dtype=dtype,
        )
        start_times = level["start_times"].to(device=birth_rates.device, dtype=dtype)
        end_times = level["end_times"].to(device=birth_rates.device, dtype=dtype)
        edge_matrix = level["edge_matrix"].to(device=birth_rates.device, dtype=dtype)
        transition_matrix = level.get("transition_matrix")
        if transition_matrix is not None:
            transition_matrix = transition_matrix.to(
                device=birth_rates.device,
                dtype=dtype,
            )

        num_samples = sampling_state_counts.sum(dim=1)
        sampling_loglik = sampling_state_counts @ log_sampling
        state_matches_samples = sampling_state_counts == num_samples[:, None]
        fixed_states = level.get("fixed_states")
        if fixed_states is not None:
            fixed_states = fixed_states.to(device=birth_rates.device, dtype=torch.long)
            state_indices = torch.arange(
                birth_rates.shape[0], device=birth_rates.device
            )
            state_matches_samples &= (fixed_states[:, None] < 0) | (
                fixed_states[:, None] == state_indices[None, :]
            )
        level_ll = (
            num_births[:, None] * log_birth[None, :]
            + sampling_loglik[:, None]
        )
        level_ll = torch.where(
            state_matches_samples,
            level_ll,
            torch.full_like(level_ll, -torch.inf),
        )

        epoch_start = epoch_times[i]
        child_elapsed = end_times[:, None] - epoch_start
        parent_elapsed = start_times[:, None] - epoch_start
        edge_ll = -rates[None, :] * (start_times[:, None] - end_times[:, None])
        edge_ll += 2.0 * (
            torch.log(
                1.0 - Cs[i, :] * torch.exp(-rates[None, :] * child_elapsed)
            )
            - torch.log(
                1.0 - Cs[i, :] * torch.exp(-rates[None, :] * parent_elapsed)
            )
        )

        level_ll += edge_matrix @ edge_ll

        if i > 0: 
            child_messages = torch.logsumexp(
                log_probabilities.unsqueeze(0) + previous_ll.unsqueeze(1),
                dim=-1,
            )
            level_ll += transition_matrix @ child_messages

        level_lls.append(level_ll)

        previous_ll = level_ll
    
    return level_ll




