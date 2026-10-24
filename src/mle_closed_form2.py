# mle_closed_form.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional

import networkx as nx

import torch
import torch.nn as nn

from src.dtos import (
    LikelihoodResult,
    UResult,
    OptimizationResult,
)

from src.closed_form_coord2 import (
    calculate_Us,
    calculate_log_likelihood,
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
        free_edges: list[tuple[int, int]],
        given_transition_times: list[float],
        device: torch.device,
        initial_birth_rates=None,
        initial_death_rates=None,
        given_death_rates=None,
        initial_sampling_probabilities=None,
        given_sampling_probabilities=None,
        initial_transition_probabilities=None,
        max_birth_rate: float = 1000.0,
        max_death_rate: float = 500.0,
        shared_sampling_rate: bool = False,
    ):
        super().__init__()

        self.num_states = num_states
        self.free_edges = free_edges
        self.shared_sampling_rate = shared_sampling_rate
        self.transition_times = torch.as_tensor(
            given_transition_times,
            dtype=torch.float64,
            device=device,
        )
        self.device_ = device

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

        if given_sampling_probabilities is not None:
            if shared_sampling_rate:
                raise ValueError(
                    "given_sampling_probabilities cannot be combined with "
                    "shared_sampling_rate."
                )
            fixed_sampling = torch.as_tensor(
                given_sampling_probabilities,
                dtype=torch.get_default_dtype(),
                device=device,
            )
            if fixed_sampling.numel() != num_states:
                raise ValueError(
                    "given_sampling_probabilities must have length num_states"
                )
            if not torch.isfinite(fixed_sampling).all() or torch.any(
                (fixed_sampling < 0.0) | (fixed_sampling > 1.0)
            ):
                raise ValueError(
                    "given_sampling_probabilities must be finite values in [0, 1]"
                )
            self.register_buffer("fixed_sampling_probabilities", fixed_sampling)
        else:
            initial_sampling = torch.as_tensor(
                initial_sampling_probabilities,
                dtype=torch.get_default_dtype(),
                device=device,
            )

            if initial_sampling.numel() != num_states:
                raise ValueError(
                    "initial_sampling_probabilities must " "have length num_states"
                )

            initial_sampling = torch.clamp(
                initial_sampling,
                1e-8,
                1.0 - 1e-8,
            )

            if shared_sampling_rate:
                initial_shared_sampling = initial_sampling.mean().reshape(1)
                self.sampling_raw = nn.Parameter(
                    inverse_sigmoid(initial_shared_sampling)
                )
            else:
                self.sampling_raw = nn.Parameter(inverse_sigmoid(initial_sampling))

        # ====================================================
        # TRANSITION PROBABILITIES
        # ====================================================

        if initial_transition_probabilities is None:

            initial_transition_probabilities = [
                [1.0 if i == j else 0.0 for j in range(num_states)]
                for i in range(num_states)
            ]

            for i, j in free_edges:
                initial_transition_probabilities[i][j] = 1.0

        initial_Q = torch.as_tensor(
            initial_transition_probabilities,
            dtype=torch.get_default_dtype(),
            device=device,
        )

        if initial_Q.shape != (
            num_states,
            num_states,
        ):
            raise ValueError(
                "initial_transition_probabilities must "
                "have shape "
                "(num_states, num_states)"
            )

        # ----------------------------------------------------
        # Raw transition parameters.
        #
        # First num_states:
        #     diagonal entries
        #
        # Remaining:
        #     free edges
        #
        # These raw values are what PyTorch optimizes.
        # ----------------------------------------------------

        transition_raw = []

        # Diagonal
        for i in range(num_states):

            transition_raw.append(
                torch.log(
                    torch.clamp(
                        initial_Q[i, i],
                        min=1e-12,
                    )
                )
            )

        # Free edges
        for i, j in free_edges:

            transition_raw.append(
                torch.log(
                    torch.clamp(
                        initial_Q[i, j],
                        min=1e-12,
                    )
                )
            )

        self.transition_raw = nn.Parameter(torch.stack(transition_raw))

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
        if hasattr(self, "fixed_sampling_probabilities"):
            return self.fixed_sampling_probabilities

        probabilities = torch.sigmoid(self.sampling_raw)
        if self.shared_sampling_rate:
            return probabilities.expand(self.num_states)
        return probabilities

    # ========================================================
    # TRANSITION PROBABILITIES
    # ========================================================

    @property
    def transition_probabilities(
        self,
    ) -> torch.Tensor:

        S = self.num_states

        diag_raw = self.transition_raw[:S]
        edge_raw = self.transition_raw[S:]

        rows = []

        for i in range(S):

            logits = [diag_raw[i]]

            columns = [i]

            for k, (u, v) in enumerate(self.free_edges):

                if u == i:

                    logits.append(edge_raw[k])

                    columns.append(v)

            probabilities = torch.softmax(
                torch.stack(logits),
                dim=0,
            )

            columns = torch.tensor(
                columns,
                dtype=torch.long,
                device=self.device_,
            )

            row = torch.zeros(
                S,
                dtype=probabilities.dtype,
                device=self.device_,
            )

            row = row.scatter(
                0,
                columns,
                probabilities,
            )

            rows.append(row)

        return torch.stack(
            rows,
            dim=0,
        )


# ============================================================
# LOG LIKELIHOOD
# ============================================================


def calculate_total_log_likelihood(
    params: PhylodynamicParameters,
    trees: list[nx.DiGraph],
    datastructures: list[dict],
    static_transition_times: list[float] | None = None,
    normalize_root_likelihood: bool = False,
) -> torch.Tensor:
    """
    Calculate the total log likelihood across all trees.
    """

    device = params.device_

    birth_rates = params.birth_rates
    death_rates = params.death_rates
    sampling_probabilities = params.sampling_probabilities
    transition_probabilities = params.transition_probabilities
    transition_times = params.transition_times

    if static_transition_times is None:
        static_transition_times = transition_times.detach().cpu().tolist()

    # --------------------------------------------------------
    # U / flow_U / Cs
    # --------------------------------------------------------

    root_times = [
        float(tree.nodes[tree.graph["root"]]["T"])
        for tree in trees
    ]
    U_result = calculate_Us(
        birth_rates,
        death_rates,
        sampling_probabilities,
        transition_probabilities,
        transition_times,
        root_times=root_times,
    )

    # print(f"U_result: {U_result}")

    # --------------------------------------------------------
    # TREE LIKELIHOOD
    # --------------------------------------------------------

    LL = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    for tree_index, (tree, datastructure) in enumerate(zip(trees, datastructures)):
        result = calculate_log_likelihood(
            datastructure,
            birth_rates,
            death_rates,
            transition_probabilities,
            sampling_probabilities,
            U_result.Cs,
            static_transition_times,
        )

        LL = LL + result[0, 0]
        if normalize_root_likelihood:
            LL = LL - torch.log1p(-U_result.root_U[tree_index, 0])

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

    transitions = params.transition_probabilities.detach().cpu().numpy()

    times = params.transition_times.detach().cpu().numpy()

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

    logfile.write("transitions:\n")

    for i, j in params.free_edges:

        logfile.write(f"{i}->{j} " f"{transitions[i, j]:.15g}\n")

    logfile.write("transition_times:\n")

    for t in times:

        logfile.write(f"{t:.15g} ")

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
        datastructures,
        normalize_root_likelihood: bool = False,
    ):
        super().__init__()

        self.params = params
        self.trees = trees
        self.datastructures = datastructures
        self.normalize_root_likelihood = normalize_root_likelihood
        self.static_transition_times = params.transition_times.detach().cpu().tolist()

    def forward(self):

        return calculate_total_log_likelihood(
            params=self.params,
            trees=self.trees,
            datastructures=self.datastructures,
            static_transition_times=self.static_transition_times,
            normalize_root_likelihood=self.normalize_root_likelihood,
        )


def optimize_parameters(
    trees: list[nx.DiGraph],
    datastructure: list[dict],
    num_states: int,
    given_transition_times: list[float],
    free_edges: list[tuple[int, int]],
    logfile: Optional[str] = None,
    initial_birth_rates=None,
    initial_death_rates=None,
    given_death_rates=None,
    initial_sampling_probabilities=None,
    initial_transition_probabilities=None,
    given_birth_rates=None,
    given_sampling_probabilities=None,
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
    shared_sampling_rate: bool = False,
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
    # PARAMETERS
    # ========================================================

    params = PhylodynamicParameters(
        num_states=num_states,
        free_edges=free_edges,
        device=device,
        initial_birth_rates=(initial_birth_rates),
        initial_death_rates=(initial_death_rates),
        given_death_rates=(given_death_rates),
        initial_sampling_probabilities=(initial_sampling_probabilities),
        given_sampling_probabilities=(given_sampling_probabilities),
        initial_transition_probabilities=(initial_transition_probabilities),
        given_transition_times=(given_transition_times),
        shared_sampling_rate=shared_sampling_rate,
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
            datastructures=datastructure,
            normalize_root_likelihood=normalize_root_likelihood,
        )

        return OptimizationResult(
            log_likelihood=LL.detach(),
            birth_rates=params.birth_rates.detach(),
            death_rates=params.death_rates.detach(),
            sampling_probabilities=(params.sampling_probabilities.detach()),
            transition_probabilities=(params.transition_probabilities.detach()),
            transition_times=(params.transition_times.detach()),
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
            datastructures=datastructure[start : start + chunk_size],
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
                datastructures=datastructure,
                static_transition_times=given_transition_times,
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
            datastructures=datastructure,
            static_transition_times=(given_transition_times),
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
        transition_probabilities=(params.transition_probabilities.detach().clone()),
        transition_times=(params.transition_times.detach().clone()),
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

    print("transition probabilities:")

    print(result.transition_probabilities.cpu().numpy())

    print(
        "transition times:",
        result.transition_times.cpu().numpy(),
    )

    return result
