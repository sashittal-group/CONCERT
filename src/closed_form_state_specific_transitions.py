from __future__ import annotations

from typing import Sequence

import torch

from src.dtos import UResult


def _safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    has_finite_value = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe_values = torch.where(has_finite_value, values, torch.zeros_like(values))
    reduced = torch.logsumexp(safe_values, dim=dim)
    return torch.where(
        has_finite_value.squeeze(dim),
        reduced,
        torch.full_like(reduced, -torch.inf),
    )


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
    if transition_times.ndim != 1 or transition_times.numel() == 0:
        raise ValueError("transition_times must be a non-empty one-dimensional tensor")
    if transition_probabilities.shape != (
        transition_times.numel(),
        birth_rates.numel(),
        birth_rates.numel(),
    ):
        raise ValueError(
            "transition_probabilities must have shape "
            "(num_epochs, num_states, num_states)"
        )

    root_times = torch.as_tensor(
        root_times,
        dtype=birth_rates.dtype,
        device=birth_rates.device,
    )
    if root_times.ndim != 1 or root_times.numel() == 0:
        raise ValueError("root_times must be a non-empty one-dimensional sequence")

    U0 = 1.0 - sampling_probabilities
    C0 = (U0 - death_rates / birth_rates) / (U0 - 1.0)
    U_values = [U0]
    flow_U_values = [torch.zeros_like(U0)]
    C_values = [C0]

    rates = birth_rates - death_rates
    for epoch_index, transition_time in enumerate(transition_times):
        dt = transition_time
        if epoch_index > 0:
            dt = transition_time - transition_times[epoch_index - 1]

        exponential = torch.exp(-rates * dt)
        denominator = 1.0 - C_values[epoch_index] * exponential
        flow_U = 1.0 - rates / (birth_rates * denominator)
        U = transition_probabilities[epoch_index] @ flow_U
        C = (U - death_rates / birth_rates) / (U - 1.0)

        flow_U_values.append(flow_U)
        U_values.append(U)
        C_values.append(C)

    U = torch.stack(U_values)
    flow_Us = torch.stack(flow_U_values)
    Cs = torch.stack(C_values)

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
        birth_rates[None, :]
        * (1.0 - root_epoch_Cs * root_exponential)
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
) -> torch.Tensor:
    num_states = birth_rates.numel()
    num_epochs = len(transition_times)
    if transition_probabilities.shape != (num_epochs, num_states, num_states):
        raise ValueError(
            "transition_probabilities must have shape "
            "(num_epochs, num_states, num_states)"
        )
    if len(datastructure) != num_epochs + 1:
        raise ValueError(
            "Expected one tree-data level before the first transition and "
            "one additional level per transition epoch."
        )
    if Cs.shape != (num_epochs + 1, num_states):
        raise ValueError("Cs must have shape (num_epochs + 1, num_states)")

    dtype = birth_rates.dtype
    device = birth_rates.device
    log_birth = torch.log(birth_rates)
    log_sampling = torch.log(
        sampling_probabilities.clamp_min(torch.finfo(dtype).tiny)
    )
    transition_times_tensor = torch.as_tensor(
        transition_times,
        dtype=dtype,
        device=device,
    )
    log_probabilities = torch.where(
        transition_probabilities > 0.0,
        torch.log(
            transition_probabilities.clamp_min(torch.finfo(dtype).tiny)
        ),
        torch.full_like(transition_probabilities, -torch.inf),
    )
    rates = birth_rates - death_rates
    epoch_times = torch.cat((transition_times_tensor.new_zeros(1), transition_times_tensor))
    previous_level: torch.Tensor | None = None

    for level_index, level in enumerate(reversed(datastructure)):
        num_births = level["num_births"].to(device=device, dtype=dtype)
        sampling_state_counts = level["num_sampling_by_state"].to(
            device=device,
            dtype=dtype,
        )
        start_times = level["start_times"].to(device=device, dtype=dtype)
        end_times = level["end_times"].to(device=device, dtype=dtype)
        edge_matrix = level["edge_matrix"].to(device=device, dtype=dtype)
        component_matrix = level.get("transition_matrix")
        if component_matrix is not None:
            component_matrix = component_matrix.to(device=device, dtype=dtype)

        num_samples = sampling_state_counts.sum(dim=1)
        sampling_log_likelihood = sampling_state_counts @ log_sampling
        state_matches_samples = sampling_state_counts == num_samples[:, None]
        level_likelihood = (
            num_births[:, None] * log_birth[None, :]
            + sampling_log_likelihood[:, None]
        )
        level_likelihood = torch.where(
            state_matches_samples,
            level_likelihood,
            torch.full_like(level_likelihood, -torch.inf),
        )

        epoch_start = epoch_times[level_index]
        child_elapsed = end_times[:, None] - epoch_start
        parent_elapsed = start_times[:, None] - epoch_start
        edge_likelihood = -rates[None, :] * (
            start_times[:, None] - end_times[:, None]
        )
        edge_likelihood += 2.0 * (
            torch.log(
                1.0
                - Cs[level_index, :]
                * torch.exp(-rates[None, :] * child_elapsed)
            )
            - torch.log(
                1.0
                - Cs[level_index, :]
                * torch.exp(-rates[None, :] * parent_elapsed)
            )
        )
        level_likelihood += edge_matrix @ edge_likelihood

        if level_index > 0:
            if previous_level is None or component_matrix is None:
                raise ValueError(
                    "A transition hierarchy matrix is required between epochs."
                )
            child_messages = _safe_logsumexp(
                log_probabilities[level_index - 1][None, :, :]
                + previous_level[:, None, :],
                dim=-1,
            )
            connected_children = component_matrix > 0
            child_contributions = torch.where(
                connected_children[:, :, None],
                child_messages[None, :, :],
                torch.zeros_like(child_messages[None, :, :]),
            )
            level_likelihood += (
                child_contributions * component_matrix[:, :, None]
            ).sum(dim=1)

        previous_level = level_likelihood

    if previous_level is None:
        raise ValueError("The tree data structure contains no levels.")
    return previous_level
