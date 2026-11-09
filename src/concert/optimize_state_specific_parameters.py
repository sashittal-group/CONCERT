from __future__ import annotations

from pathlib import Path
import time

import networkx as nx
import torch
from torch import nn

from src.concert.closed_form_state_specific_transitions import (
    calculate_Us,
    calculate_log_likelihood,
)
from src.concert.tree_types import OptimizationResult
from src.concert.optimize_tree_parameters import PhylodynamicParameters


STATE_NAMES = ("NC", "PT", "LN")
EPOCH_EDGES = ((0, 1), (1, 2))
TRANSITION_PROBABILITY_EPSILON = 1e-8


class StateSpecificTransitionParameters(PhylodynamicParameters):
    def __init__(
        self,
        transition_times: list[float],
        device: torch.device,
        *,
        epoch_edges: tuple[tuple[int, int], ...] | None = None,
        num_states: int = len(STATE_NAMES),
        initial_transition_probabilities: tuple[float, ...] | None = None,
        initial_birth_rates=None,
        initial_death_rates=None,
        initial_sampling_probabilities=None,
        given_sampling_probabilities=None,
    ) -> None:
        if epoch_edges is None:
            epoch_edges = EPOCH_EDGES
        if num_states < 2:
            raise ValueError("At least two states are required.")
        if not transition_times:
            raise ValueError("At least one state-specific transition time is required.")
        if len(transition_times) != len(epoch_edges):
            raise ValueError(
                "Each state-specific transition time must have one epoch edge."
            )
        if any(time_value < 0.0 for time_value in transition_times):
            raise ValueError("Transition times must be nonnegative.")
        if any(
            source < 0
            or source >= num_states
            or target < 0
            or target >= num_states
            or source == target
            for source, target in epoch_edges
        ):
            raise ValueError(
                "Each epoch edge must connect two distinct valid states."
            )
        if any(
            not transition_times[index - 1] < transition_times[index]
            for index in range(1, len(transition_times))
        ):
            raise ValueError("Transition times must be strictly increasing.")
        source_states = tuple(dict.fromkeys(source for source, _ in epoch_edges))
        if initial_transition_probabilities is None:
            initial_transition_probabilities = (0.25,) * len(source_states)
        if len(initial_transition_probabilities) != len(source_states) or any(
            not 0.0 < probability < 1.0
            for probability in initial_transition_probabilities
        ):
            raise ValueError(
                "Initial transition probabilities must match the unique source "
                "states and be in (0, 1)."
            )

        identity = torch.eye(num_states, dtype=torch.float64, device=device)
        super().__init__(
            num_states=num_states,
            free_edges=list(epoch_edges),
            given_transition_times=transition_times,
            device=device,
            initial_birth_rates=initial_birth_rates,
            initial_death_rates=initial_death_rates,
            initial_sampling_probabilities=initial_sampling_probabilities,
            given_sampling_probabilities=given_sampling_probabilities,
            initial_transition_probabilities=identity.tolist(),
        )

        self.transition_raw.requires_grad_(False)
        self.epoch_edges = epoch_edges
        self.source_states = source_states
        initial = torch.as_tensor(
            initial_transition_probabilities,
            dtype=torch.float64,
            device=device,
        )
        self.source_transition_raw = nn.Parameter(torch.logit(initial))

    @property
    def transition_probabilities(self) -> torch.Tensor:
        probabilities = TRANSITION_PROBABILITY_EPSILON + (
            1.0 - 2.0 * TRANSITION_PROBABILITY_EPSILON
        ) * torch.sigmoid(self.source_transition_raw)
        probability_by_source = dict(zip(self.source_states, probabilities))
        identity = torch.eye(
            self.num_states,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        matrices = []
        for source, target in self.epoch_edges:
            probability = probability_by_source[source]
            rows = []
            for state in range(self.num_states):
                if state != source:
                    rows.append(identity[state])
                    continue
                rows.append(
                    torch.stack(
                        [
                            1.0 - probability
                            if destination == source
                            else probability
                            if destination == target
                            else torch.zeros_like(probability)
                            for destination in range(self.num_states)
                        ]
                    )
                )
            matrices.append(torch.stack(rows))
        return torch.stack(matrices)


def calculate_total_log_likelihood(
    params: StateSpecificTransitionParameters,
    trees: list[nx.DiGraph],
    datastructures: list[list[dict]],
    *,
    normalize_root_likelihood: bool = False,
) -> torch.Tensor:
    if len(trees) != len(datastructures):
        raise ValueError("Each tree must have one matching likelihood data structure.")

    transition_probabilities = params.transition_probabilities
    u_result = calculate_Us(
        params.birth_rates,
        params.death_rates,
        params.sampling_probabilities,
        transition_probabilities,
        params.transition_times,
        root_times=[
            float(tree.nodes[tree.graph["root"]]["T"])
            for tree in trees
        ],
    )

    total_log_likelihood = torch.zeros(
        (),
        dtype=params.birth_rates.dtype,
        device=params.device_,
    )
    transition_times = params.transition_times.detach().cpu().tolist()
    for tree_index, datastructure in enumerate(datastructures):
        local_epoch_count = len(datastructure) - 1
        tree_likelihood = calculate_log_likelihood(
            datastructure,
            params.birth_rates,
            params.death_rates,
            transition_probabilities[:local_epoch_count],
            params.sampling_probabilities,
            u_result.Cs[:local_epoch_count + 1],
            transition_times[:local_epoch_count],
        )
        total_log_likelihood = total_log_likelihood + tree_likelihood[0, 0]
        if normalize_root_likelihood:
            total_log_likelihood = total_log_likelihood - torch.log1p(
                -u_result.root_U[tree_index, 0]
            )
    return total_log_likelihood


def optimize_parameters(
    trees: list[nx.DiGraph],
    datastructures: list[list[dict]],
    transition_times: list[float],
    *,
    epoch_edges: tuple[tuple[int, int], ...] = EPOCH_EDGES,
    num_states: int = len(STATE_NAMES),
    logfile: str | Path | None = None,
    device: torch.device | str = "cpu",
    initial_transition_probabilities: tuple[float, ...] | None = None,
    initial_birth_rates=None,
    initial_death_rates=None,
    initial_sampling_probabilities=None,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    normalize_root_likelihood: bool = False,
) -> OptimizationResult:
    if lbfgs_max_iter < 1:
        raise ValueError("lbfgs_max_iter must be at least one.")
    if not trees:
        raise ValueError("At least one tree is required.")
    if len(trees) != len(datastructures):
        raise ValueError("Each tree must have one matching likelihood data structure.")

    optimization_device = torch.device(device)
    params = StateSpecificTransitionParameters(
        transition_times,
        optimization_device,
        epoch_edges=epoch_edges,
        num_states=num_states,
        initial_transition_probabilities=initial_transition_probabilities,
        initial_birth_rates=initial_birth_rates,
        initial_death_rates=initial_death_rates,
        initial_sampling_probabilities=initial_sampling_probabilities,
    ).to(optimization_device)
    optimizer = torch.optim.LBFGS(
        [parameter for parameter in params.parameters() if parameter.requires_grad],
        lr=1.0,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    log_handle = None
    if logfile is not None:
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", buffering=1)

    best_log_likelihood = torch.tensor(
        -torch.inf,
        dtype=params.birth_raw.dtype,
        device=optimization_device,
    )
    best_state: dict[str, torch.Tensor] | None = None
    closure_calls = 0
    started = time.perf_counter()

    def closure() -> torch.Tensor:
        nonlocal best_log_likelihood, best_state, closure_calls
        optimizer.zero_grad(set_to_none=True)
        log_likelihood = calculate_total_log_likelihood(
            params,
            trees,
            datastructures,
            normalize_root_likelihood=normalize_root_likelihood,
        )
        if not torch.isfinite(log_likelihood):
            raise FloatingPointError("Non-finite likelihood during LBFGS optimization.")
        loss = -log_likelihood
        loss.backward()
        closure_calls += 1
        with torch.no_grad():
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood.detach().clone()
                best_state = {
                    name: value.detach().clone()
                    for name, value in params.state_dict().items()
                }
            if log_handle is not None and closure_calls % 10 == 0:
                log_handle.write(
                    f"ITER {closure_calls} LL={log_likelihood.item():.15g}\n"
                )
        return loss

    try:
        optimizer.step(closure)
        if best_state is None:
            raise RuntimeError("LBFGS completed without evaluating a finite fit.")
        params.load_state_dict(best_state)
        with torch.no_grad():
            final_log_likelihood = calculate_total_log_likelihood(
                params,
                trees,
                datastructures,
                normalize_root_likelihood=normalize_root_likelihood,
            )
            result = OptimizationResult(
                log_likelihood=final_log_likelihood.detach(),
                birth_rates=params.birth_rates.detach().clone(),
                death_rates=params.death_rates.detach().clone(),
                sampling_probabilities=params.sampling_probabilities.detach().clone(),
                transition_probabilities=params.transition_probabilities.detach().clone(),
                transition_times=params.transition_times.detach().clone(),
            )
        if log_handle is not None:
            log_handle.write(
                f"FINAL LL={result.log_likelihood.item():.15g} "
                f"elapsed_seconds={time.perf_counter() - started:.3f}\n"
            )
        return result
    finally:
        if log_handle is not None:
            log_handle.close()
