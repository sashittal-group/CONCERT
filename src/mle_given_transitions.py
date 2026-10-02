# mle_closed_form.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.dtos import (
    PreparedTree,
    LikelihoodResult,
    UResult,
    OptimizationResult,
)

from src.closed_form_coord import (
    calculate_Us,
    calculate_log_likelihood_levelwise,
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

# torch.set_default_dtype(torch.float64)


# ============================================================
# DEVICE
# ============================================================


def get_device() -> torch.device:
    """
    Prefer CUDA, then Apple MPS, then CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed_log(label: str, start: float, device: torch.device) -> None:
    _sync_device(device)
    elapsed = time.perf_counter() - start
    print(f"[timing] {label}: {elapsed:.3f}s", flush=True)


# ============================================================
# HELPERS
# ============================================================


def inverse_sigmoid(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Inverse of sigmoid.

    Maps x in (0,1) to (-inf,+inf).
    """

    x = torch.clamp(x, eps, 1.0 - eps)

    return torch.log(x) - torch.log1p(-x)


def as_tensor(
    x,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert Python/numpy/tensor input to a tensor.
    """

    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(
        x,
        device=device,
        dtype=dtype,
    )


def move_tree_to_device(
    tree: PreparedTree,
    device: torch.device,
) -> PreparedTree:
    """
    Move all tensor fields in PreparedTree to the requested device.
    """

    return PreparedTree(
        root=tree.root,
        original_id=tree.original_id,
        time=tree.time,
        state=tree.state,
        parent=tree.parent,
        children=tree.children,
        edge_idx_to_parent=tree.edge_idx_to_parent,
        edge_parent=tree.edge_parent,
        edge_child=tree.edge_child,
        postorder=tree.postorder,
        levels=tree.levels,
        level_batches=tree.level_batches,
        time_t=tree.time_t.to(device),
    )


import torch
import torch.nn as nn


def inverse_sigmoid(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    x = torch.clamp(
        x,
        eps,
        1.0 - eps,
    )

    return torch.log(x / (1.0 - x))


class PhylodynamicParameters(nn.Module):

    def __init__(
        self,
        num_states: int,
        device: torch.device,
        initial_birth_rates=None,
        initial_death_rates=None,
        given_death_rates=None,
        initial_sampling_probabilities=None,
        same_sampling_probability_for_all_states: bool = False,
        max_birth_rate: float = 20.0,
        max_death_rate: float = 10.0,
    ):
        super().__init__()

        self.num_states = num_states
        self.device_ = device
        self.same_sampling_probability_for_all_states = (
            same_sampling_probability_for_all_states
        )

        self.max_birth_rate = max_birth_rate
        self.max_death_rate = max_death_rate

        if max_birth_rate <= 0:
            raise ValueError("max_birth_rate must be > 0")

        if max_death_rate < 0:
            raise ValueError("max_death_rate must be >= 0")

        if max_death_rate > max_birth_rate:
            raise ValueError("max_death_rate must be <= max_birth_rate")

        # ====================================================
        # INITIAL VALUES
        # ====================================================

        if initial_birth_rates is None:
            initial_birth_rates = [5.0] * num_states

        if initial_death_rates is None and given_death_rates is None:
            initial_death_rates = [1.0] * num_states

        if given_death_rates is not None:
            initial_death_rates = given_death_rates

        if initial_sampling_probabilities is None:
            initial_sampling_probabilities = [0.5] * num_states

        # ====================================================
        # BIRTH / DEATH
        # ====================================================

        initial_birth = torch.as_tensor(
            initial_birth_rates,
            dtype=torch.get_default_dtype(),
            device=device,
        )

        initial_death = torch.as_tensor(
            initial_death_rates,
            dtype=torch.get_default_dtype(),
            device=device,
        )

        if initial_birth.numel() != num_states:
            raise ValueError("initial_birth_rates must have " "length num_states")

        if initial_death.numel() != num_states:
            raise ValueError("initial_death_rates must have " "length num_states")

        if torch.any(initial_birth < initial_death):
            raise ValueError("initial_birth_rates must satisfy " "birth >= death")

        if torch.any(initial_birth > max_birth_rate):
            raise ValueError("initial_birth_rates must be <= " "max_birth_rate")

        if torch.any(initial_death > max_death_rate):
            raise ValueError("initial_death_rates must be <= " "max_death_rate")

        if given_death_rates is None:
            # death = max_death_rate * sigmoid(death_raw)
            death_fraction = initial_death / max_death_rate
            death_fraction = torch.clamp(
                death_fraction,
                1e-8,
                1.0 - 1e-8,
            )
            self.death_raw = nn.Parameter(inverse_sigmoid(death_fraction))
        else:
            self.register_buffer("fixed_death_rates", initial_death.clone())

        # ----------------------------------------------------
        # birth =
        # death + (max_birth_rate - death) * sigmoid(birth_raw)
        #
        # This guarantees:
        #     death <= birth <= max_birth_rate
        # ----------------------------------------------------

        birth_fraction = (initial_birth - initial_death) / (
            max_birth_rate - initial_death
        )

        birth_fraction = torch.clamp(
            birth_fraction,
            1e-8,
            1.0 - 1e-8,
        )

        self.birth_raw = nn.Parameter(inverse_sigmoid(birth_fraction))

        # ====================================================
        # SAMPLING PROBABILITIES
        # ====================================================

        initial_sampling = torch.as_tensor(
            initial_sampling_probabilities,
            dtype=torch.get_default_dtype(),
            device=device,
        )

        if initial_sampling.numel() != num_states:
            raise ValueError(
                "initial_sampling_probabilities must " "have length num_states"
            )

        if same_sampling_probability_for_all_states:
            initial_sampling = initial_sampling.mean().reshape(1)

        initial_sampling = torch.clamp(
            initial_sampling,
            1e-8,
            1.0 - 1e-8,
        )

        self.sampling_raw = nn.Parameter(inverse_sigmoid(initial_sampling))

    # ========================================================
    # BIRTH RATES
    # ========================================================

    @property
    def birth_rates(self) -> torch.Tensor:

        death = self.death_rates

        return death + (self.max_birth_rate - death) * torch.sigmoid(self.birth_raw)

    # ========================================================
    # DEATH RATES
    # ========================================================

    @property
    def death_rates(self) -> torch.Tensor:
        if hasattr(self, "fixed_death_rates"):
            return self.fixed_death_rates

        return self.max_death_rate * torch.sigmoid(self.death_raw)

    # ========================================================
    # SAMPLING PROBABILITIES
    # ========================================================

    @property
    def sampling_probabilities(
        self,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(self.sampling_raw)
        if self.same_sampling_probability_for_all_states:
            return probabilities.expand(self.num_states)
        return probabilities


# ============================================================
# LOG LIKELIHOOD
# ============================================================


def calculate_total_log_likelihood(
    params: PhylodynamicParameters,
    trees: list,
    root_time: float = 1.0,
    normalize_root_likelihood: bool = False,
) -> torch.Tensor:
    """
    Calculate the total log likelihood across all trees.
    """

    device = params.device_

    birth_rates = params.birth_rates
    death_rates = params.death_rates
    sampling_probabilities = params.sampling_probabilities

    # --------------------------------------------------------
    # U / flow_U / Cs
    # --------------------------------------------------------

    U0 = 1.0 - sampling_probabilities
    numerator = U0 - death_rates / birth_rates
    denominator = U0 - 1.0
    C0 = numerator / denominator

    r = birth_rates - death_rates
    exponential = torch.exp(-r * root_time)
    denominator = 1.0 - C0 * exponential
    U_root = 1.0 - r / (birth_rates * denominator)

    # --------------------------------------------------------
    # TREE LIKELIHOOD
    # --------------------------------------------------------

    LL = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    for internal_node_state_counts, leaf_state_counts, state_edges in trees:

        for state in range(params.num_states):
            LL = (
                LL
                + leaf_state_counts[state] * torch.log(sampling_probabilities[state])
                + internal_node_state_counts[state] * torch.log(birth_rates[state])
            )

            for s_t, e_t in state_edges[state]:
                LL = (
                    LL
                    - (s_t - e_t) * (birth_rates[state] - death_rates[state])
                    + +2.0
                    * (
                        torch.log(
                            1.0
                            - C0[state]
                            * torch.exp(
                                -(birth_rates[state] - death_rates[state]) * e_t
                            )
                        )
                        - torch.log(
                            1.0
                            - C0[state]
                            * torch.exp(
                                -(birth_rates[state] - death_rates[state]) * s_t
                            )
                        )
                    )
                )

        if normalize_root_likelihood:
            LL = LL - torch.log1p(-U_root[0])

        # print(f"Result for tree: {result.LL}")

    # --------------------------------------------------------
    # PRIOR
    # --------------------------------------------------------

    # if prior_fn is not None:

    #     prior_value = prior_fn(transition_times)

    #     prior_value = as_tensor(
    #         prior_value,
    #         device=device,
    #     )

    #     LL = LL + prior_value

    return LL


# ============================================================
# LOGGING
# ============================================================


def log_iteration(
    step: int,
    LL: torch.Tensor,
    params: PhylodynamicParameters,
    logfile,
) -> None:

    LL_value = LL.detach().item()

    birth = params.birth_rates.detach().cpu().numpy()

    death = params.death_rates.detach().cpu().numpy()

    sampling = params.sampling_probabilities.detach().cpu().numpy()

    print(f"ITER {step}  LL = {LL_value:.15f}")

    if logfile is None:
        return

    logfile.write(f"ITER {step} LL={LL_value:.15f}\n")

    logfile.write("birth: ")
    logfile.write(" ".join(f"{x:.15g}" for x in birth))
    logfile.write("\n")

    logfile.write("death: ")
    logfile.write(" ".join(f"{x:.15g}" for x in death))
    logfile.write("\n")

    logfile.write("sampling: ")
    logfile.write(" ".join(f"{x:.15g}" for x in sampling))
    logfile.write("\n")

    logfile.write("\n\n")
    logfile.flush()


# ============================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================


class LikelihoodModel(nn.Module):

    def __init__(
        self,
        params,
        trees,
        normalize_root_likelihood: bool = False,
    ):
        super().__init__()

        self.params = params
        self.trees = trees
        self.normalize_root_likelihood = normalize_root_likelihood

    def forward(self):

        return calculate_total_log_likelihood(
            params=self.params,
            trees=self.trees,
            normalize_root_likelihood=self.normalize_root_likelihood,
        )


def optimize_parameters(
    trees: list,
    num_states: int,
    logfile: Optional[str] = None,
    initial_birth_rates=None,
    initial_death_rates=None,
    given_death_rates=None,
    initial_sampling_probabilities=None,
    same_sampling_probability_for_all_states: bool = True,
    # --------------------------------------------------------
    # PyTorch optimizer settings
    # --------------------------------------------------------
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    max_steps: int = 5000,
    patience: int = 500,
    # Used by LBFGS.
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    device: Optional[torch.device | str] = None,
    normalize_root_likelihood: bool = False,
):

    total_start = time.perf_counter()

    # ========================================================
    # DEVICE
    # ========================================================

    if device is None:
        device = get_device()

    else:
        device = torch.device(device)

    if patience < 1:
        raise ValueError("patience must be at least 1")

    print(f"Using device: {device}")
    print("[timing] optimization setup started", flush=True)

    # ========================================================
    # MOVE TREES
    # ========================================================

    # given_transition_epochs = [
    #     move_epoch_to_device(
    #         epoch,
    #         device,
    #     )
    #     for epoch in given_transition_epochs
    # ]

    # ========================================================
    # PARAMETERS
    # ========================================================

    params = PhylodynamicParameters(
        num_states=num_states,
        device=device,
        initial_birth_rates=(initial_birth_rates),
        initial_death_rates=(initial_death_rates),
        given_death_rates=(given_death_rates),
        initial_sampling_probabilities=(initial_sampling_probabilities),
        same_sampling_probability_for_all_states=(
            same_sampling_probability_for_all_states
        ),
    ).to(device)
    _timed_log("parameter construction", total_start, device)

    # ========================================================
    # LOG FILE
    # ========================================================

    log_file = None

    if logfile is not None:
        log_file = open(
            Path(logfile),
            "w",
            buffering=1,
        )

    # ========================================================
    # OPTIMIZABLE PARAMETERS
    # ========================================================

    trainable_parameters = [p for p in params.parameters() if p.requires_grad]

    if len(trainable_parameters) == 0:

        # No parameters to optimize.
        # Still calculate likelihood.

        LL = calculate_total_log_likelihood(
            params=params,
            trees=trees,
            normalize_root_likelihood=normalize_root_likelihood,
        )

        return OptimizationResult(
            log_likelihood=LL.detach(),
            birth_rates=params.birth_rates.detach(),
            death_rates=params.death_rates.detach(),
            sampling_probabilities=(params.sampling_probabilities.detach()),
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
        )

    elif optimizer_name == "lbfgs":

        optimizer = torch.optim.LBFGS(
            trainable_parameters,
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

    else:

        raise ValueError("optimizer_name must be " "'adam' or 'lbfgs'")

    # ========================================================
    # BEST SOLUTION
    # ========================================================

    best_LL = torch.tensor(
        -torch.inf,
        device=device,
        dtype=torch.float64,
    )

    best_state = None

    # ========================================================
    # ADAM
    # ========================================================

    compile_start = time.perf_counter()
    chunk_size = 10
    compiled_models = []

    for start in range(0, len(trees), chunk_size):
        chunk_model = LikelihoodModel(
            params=params,
            trees=trees[start : start + chunk_size],
            normalize_root_likelihood=normalize_root_likelihood,
        )
        compiled_models.append(
            torch.compile(
                chunk_model,
                fullgraph=False,
            )
        )

    _timed_log("torch.compile request returned", compile_start, device)

    if optimizer_name == "adam":

        steps_without_improvement = 0

        for step in range(1, max_steps + 1):

            step_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            forward_start = time.perf_counter()
            loss = -torch.stack(
                [compiled_model() for compiled_model in compiled_models]
            ).sum()
            if step == 1:
                _timed_log(
                    "step 1 forward (includes graph construction)",
                    forward_start,
                    device,
                )

            # Detect numerical failures early.
            if not torch.isfinite(loss):

                raise FloatingPointError(
                    f"Non-finite loss at step {step}: " f"{loss.detach().item()}"
                )

            # The forward value is already the current likelihood. Avoid
            # evaluating the complete tree likelihood a second time here.
            current_LL = -loss.detach()

            state_before_step = {
                name: value.detach().clone()
                for name, value in params.state_dict().items()
            }

            backward_start = time.perf_counter()
            loss.backward()
            if step == 1:
                _timed_log("step 1 backward", backward_start, device)

            optimizer_start = time.perf_counter()
            optimizer.step()
            if step == 1:
                _timed_log("step 1 optimizer.step", optimizer_start, device)
                _timed_log("step 1 total", step_start, device)

            # ------------------------------------------------
            # Best solution
            # ------------------------------------------------

            if current_LL > best_LL:

                best_LL = current_LL
                steps_without_improvement = 0

                # Save the parameters that produced current_LL, which was
                # computed immediately before the optimizer update.
                best_state = state_before_step

            else:
                steps_without_improvement += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if step % 10 == 0:

                log_iteration(
                    step=step,
                    LL=current_LL,
                    params=params,
                    logfile=log_file,
                )

            if steps_without_improvement >= patience:
                print(
                    f"Early stopping at step {step}: objective did not improve "
                    f"for {patience} steps.",
                    flush=True,
                )
                break

    # ========================================================
    # LBFGS
    # ========================================================

    else:

        step_counter = 0

        def closure():

            nonlocal step_counter
            nonlocal best_LL
            nonlocal best_state

            optimizer.zero_grad(set_to_none=True)

            LL = calculate_total_log_likelihood(
                params=params,
                trees=trees,
                normalize_root_likelihood=normalize_root_likelihood,
            )

            loss = -LL

            if not torch.isfinite(loss):

                raise FloatingPointError("Non-finite loss during LBFGS")

            loss.backward()

            step_counter += 1

            with torch.no_grad():

                if LL > best_LL:

                    best_LL = LL.detach()

                    best_state = {
                        name: value.detach().clone()
                        for name, value in params.state_dict().items()
                    }

                if step_counter % 10 == 0:

                    log_iteration(
                        step=step_counter,
                        LL=LL,
                        params=params,
                        logfile=log_file,
                    )

            return loss

        optimizer.step(closure)

    # ========================================================
    # RESTORE BEST SOLUTION
    # ========================================================

    if best_state is not None:

        params.load_state_dict(best_state)

    # ========================================================
    # FINAL LIKELIHOOD
    # ========================================================

    with torch.no_grad():

        final_LL = calculate_total_log_likelihood(
            params=params,
            trees=trees,
            normalize_root_likelihood=normalize_root_likelihood,
        )

    # ========================================================
    # CLOSE LOG
    # ========================================================

    if log_file is not None:
        log_file.close()

    # ========================================================
    # RESULT
    # ========================================================

    result = OptimizationResult(
        log_likelihood=final_LL.detach(),
        birth_rates=(params.birth_rates.detach().clone()),
        death_rates=(params.death_rates.detach().clone()),
        sampling_probabilities=(params.sampling_probabilities.detach().clone()),
    )

    print("\n==============================")
    print("OPTIMIZATION COMPLETE")
    print("==============================")

    print(f"Best LL = " f"{result.log_likelihood.item():.15f}")

    print(
        "birth:",
        result.birth_rates.cpu().numpy(),
    )

    print(
        "death:",
        result.death_rates.cpu().numpy(),
    )

    print(
        "sampling:",
        result.sampling_probabilities.cpu().numpy(),
    )

    return result
