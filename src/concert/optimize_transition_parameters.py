# mle_closed_form.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional

import networkx as nx

import torch
import torch.nn as nn

from src.concert.tree_types import (
    LikelihoodResult,
    UResult,
    OptimizationResult,
)

from src.concert.closed_form_transitions import (
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
        max_birth_rate: float = 10000000000.0,
        max_death_rate: float = 5000000000.0,
        shared_sampling_rate: bool = False,
        num_trees: int | None = None,
        per_tree_sampling_rates: bool = False,
        per_tree_state_sampling_rates: bool = False,
        per_tree_transition_rates: bool = False,
        tree_transition_times: list[list[float]] | None = None,
        tree_free_edges: list[list[tuple[int, int]]] | None = None,
    ):
        super().__init__()

        self.num_states = num_states
        self.free_edges = free_edges
        self.shared_sampling_rate = shared_sampling_rate
        self.per_tree_sampling_rates = per_tree_sampling_rates
        self.per_tree_state_sampling_rates = per_tree_state_sampling_rates
        self.per_tree_transition_rates = per_tree_transition_rates
        self.num_trees = num_trees
        if per_tree_transition_rates:
            if tree_free_edges is None:
                self.tree_free_edges = [list(free_edges) for _ in range(num_trees or 0)]
            elif num_trees is None or len(tree_free_edges) != num_trees:
                raise ValueError(
                    "Per-tree transition edges must be provided for every tree"
                )
            else:
                self.tree_free_edges = [list(edges) for edges in tree_free_edges]
            for edges in self.tree_free_edges:
                if len(edges) != len(set(edges)):
                    raise ValueError("Per-tree transition edges must be unique")
                if any(edge not in free_edges for edge in edges):
                    raise ValueError(
                        "Per-tree transition edges must be included in free_edges"
                    )
        elif tree_free_edges is not None:
            raise ValueError(
                "Per-tree transition edges require per-tree transition rates"
            )
        else:
            self.tree_free_edges = None
        self.transition_times = torch.as_tensor(
            given_transition_times,
            dtype=torch.float64,
            device=device,
        )
        self.device_ = device
        if per_tree_transition_rates:
            if num_trees is None or num_trees <= 0:
                raise ValueError(
                    "per-tree transition rates require a positive num_trees"
                )
            if tree_transition_times is None or len(tree_transition_times) != num_trees:
                raise ValueError(
                    "per-tree transition rates require transition times for "
                    "every tree"
                )
            if any(not times for times in tree_transition_times):
                raise ValueError(
                    "Each tree must have at least one observed transition time"
                )
            self.tree_transition_times = [
                torch.as_tensor(times, dtype=torch.float64, device=device)
                for times in tree_transition_times
            ]
        else:
            self.tree_transition_times = None

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

        if per_tree_sampling_rates or per_tree_state_sampling_rates:
            if num_trees is None or num_trees <= 0:
                raise ValueError(
                    "per-tree sampling rates require a positive num_trees"
                )
            if per_tree_sampling_rates and per_tree_state_sampling_rates:
                raise ValueError(
                    "per-tree shared-state and per-tree state-specific sampling "
                    "rates are mutually exclusive"
                )
            if shared_sampling_rate:
                raise ValueError(
                    "per-tree sampling rates cannot be combined with "
                    "shared_sampling_rate"
                )
            if given_sampling_probabilities is not None:
                raise ValueError(
                    "per-tree sampling rates cannot be combined with fixed "
                    "sampling probabilities"
                )

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
            fixed_sampling = torch.as_tensor(
                given_sampling_probabilities,
                dtype=torch.get_default_dtype(),
                device=device,
            ).reshape(-1)
            if fixed_sampling.numel() == 1:
                fixed_sampling = fixed_sampling.expand(num_states).clone()
            if fixed_sampling.numel() != num_states:
                raise ValueError(
                    "given_sampling_probabilities must be a scalar or have "
                    "length num_states"
                )
            if not torch.all(torch.isfinite(fixed_sampling)) or torch.any(
                (fixed_sampling <= 0.0) | (fixed_sampling >= 1.0)
            ):
                raise ValueError(
                    "given_sampling_probabilities must be finite and between "
                    "0 and 1"
                )
            self.register_buffer("fixed_sampling_probabilities", fixed_sampling)
        else:
            self.register_buffer("fixed_sampling_probabilities", None)
            initial_sampling = torch.as_tensor(
                initial_sampling_probabilities,
                dtype=torch.get_default_dtype(),
                device=device,
            )

            if per_tree_state_sampling_rates:
                if initial_sampling.numel() == num_states:
                    initial_sampling = initial_sampling.reshape(1, num_states).expand(
                        num_trees, -1
                    ).clone()
                elif initial_sampling.numel() == num_trees * num_states:
                    initial_sampling = initial_sampling.reshape(num_trees, num_states)
                else:
                    raise ValueError(
                        "initial sampling probabilities must contain one value "
                        "per tree and state"
                    )
            elif per_tree_sampling_rates:
                if initial_sampling.numel() == num_states:
                    initial_sampling = initial_sampling.mean().expand(num_trees).clone()
                elif initial_sampling.numel() != num_trees:
                    raise ValueError(
                        "initial sampling probabilities must contain one value "
                        "per tree"
                    )
                else:
                    initial_sampling = initial_sampling.reshape(num_trees)
            elif initial_sampling.numel() != num_states:
                raise ValueError(
                    "initial_sampling_probabilities must " "have length num_states"
                )

            initial_sampling = torch.clamp(
                initial_sampling,
                1e-8,
                1.0 - 1e-8,
            )

            if per_tree_state_sampling_rates or per_tree_sampling_rates:
                self.sampling_raw = nn.Parameter(inverse_sigmoid(initial_sampling))
            elif shared_sampling_rate:
                initial_shared_sampling = initial_sampling.mean().reshape(1)
                self.sampling_raw = nn.Parameter(inverse_sigmoid(initial_shared_sampling))
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
        for _ in range(num_trees if per_tree_transition_rates else 1):
            tree_transition_raw = [
                torch.log(torch.clamp(initial_Q[i, i], min=1e-12))
                for i in range(num_states)
            ]
            tree_transition_raw.extend(
                torch.log(torch.clamp(initial_Q[i, j], min=1e-12))
                for i, j in free_edges
            )
            transition_raw.append(torch.stack(tree_transition_raw))
        self.transition_raw = nn.Parameter(
            torch.stack(transition_raw)
            if per_tree_transition_rates
            else transition_raw[0]
        )

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
        if self.fixed_sampling_probabilities is not None:
            return self.fixed_sampling_probabilities

        probabilities = torch.sigmoid(self.sampling_raw)
        if self.per_tree_state_sampling_rates:
            return probabilities
        if self.per_tree_sampling_rates:
            return probabilities[:, None].expand(-1, self.num_states)
        if self.shared_sampling_rate:
            return probabilities.expand(self.num_states)
        return probabilities

    @property
    def tree_sampling_probabilities(self) -> torch.Tensor | None:
        if not (
            self.per_tree_sampling_rates or self.per_tree_state_sampling_rates
        ):
            return None
        return torch.sigmoid(self.sampling_raw)

    # ========================================================
    # TRANSITION PROBABILITIES
    # ========================================================

    @property
    def transition_probabilities(
        self,
    ) -> torch.Tensor:
        def build_matrix(
            raw: torch.Tensor,
            free_edges: list[tuple[int, int]],
        ) -> torch.Tensor:
            num_states = self.num_states
            diagonal = raw[:num_states]
            edge_values = raw[num_states:]
            rows = []
            for source in range(num_states):
                logits = [diagonal[source]]
                columns = [source]
                for edge_source, target in free_edges:
                    if edge_source == source:
                        edge_index = self.free_edges.index((edge_source, target))
                        logits.append(edge_values[edge_index])
                        columns.append(target)
                probabilities = torch.softmax(torch.stack(logits), dim=0)
                column_indices = torch.tensor(
                    columns,
                    dtype=torch.long,
                    device=self.device_,
                )
                row = torch.zeros(
                    num_states,
                    dtype=probabilities.dtype,
                    device=self.device_,
                ).scatter(0, column_indices, probabilities)
                rows.append(row)
            return torch.stack(rows, dim=0)

        if self.per_tree_transition_rates:
            return torch.stack(
                [
                    build_matrix(raw, self.tree_free_edges[tree_index])
                    for tree_index, raw in enumerate(self.transition_raw)
                ]
            )
        return build_matrix(self.transition_raw, self.free_edges)


# ============================================================
# LOG LIKELIHOOD
# ============================================================


def calculate_total_log_likelihood(
    params: PhylodynamicParameters,
    trees: list[nx.DiGraph],
    datastructures: list[dict],
    static_transition_times: list[float] | None = None,
    normalize_root_likelihood: bool = False,
    no_stem: bool = False,
    static_tree_transition_times: list[list[float]] | None = None,
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
    if static_tree_transition_times is None and params.tree_transition_times is not None:
        static_tree_transition_times = [
            times.detach().cpu().tolist()
            for times in params.tree_transition_times
        ]

    # --------------------------------------------------------
    # U / flow_U / Cs
    # --------------------------------------------------------

    root_times = [
        float(tree.nodes[tree.graph["root"]]["T"])
        for tree in trees
    ]
    if sampling_probabilities.ndim == 2 or transition_probabilities.ndim == 3:
        if sampling_probabilities.ndim == 2:
            expected_sampling_shape = (len(trees), params.num_states)
            if tuple(sampling_probabilities.shape) != expected_sampling_shape:
                raise ValueError(
                    "Per-tree sampling probabilities must have shape "
                    f"{expected_sampling_shape}; found "
                    f"{tuple(sampling_probabilities.shape)}"
                )
        if transition_probabilities.ndim == 3:
            expected_transition_shape = (
                len(trees),
                params.num_states,
                params.num_states,
            )
            if tuple(transition_probabilities.shape) != expected_transition_shape:
                raise ValueError(
                    "Per-tree transition probabilities must have shape "
                    f"{expected_transition_shape}; found "
                    f"{tuple(transition_probabilities.shape)}"
                )
        if params.per_tree_transition_rates and (
            static_tree_transition_times is None
            or len(static_tree_transition_times) != len(trees)
        ):
            raise ValueError(
                "Per-tree transition rates require transition times for each tree"
            )
        LL = torch.zeros((), device=device, dtype=torch.float64)
        for tree_index, (tree, datastructure) in enumerate(zip(trees, datastructures)):
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
                static_tree_transition_times[tree_index]
                if params.per_tree_transition_rates
                else static_transition_times
            )
            tree_transition_tensor = (
                params.tree_transition_times[tree_index]
                if params.per_tree_transition_rates
                else transition_times
            )
            tree_u_result = calculate_Us(
                birth_rates,
                death_rates,
                tree_sampling,
                tree_transitions,
                tree_transition_tensor,
                root_times=[root_times[tree_index]],
            )
            result = calculate_log_likelihood(
                datastructure,
                birth_rates,
                death_rates,
                tree_transitions,
                tree_sampling,
                tree_u_result.Cs,
                tree_times,
            )
            tree_log_likelihood = result[0, 0]
            if no_stem:
                tree_log_likelihood = crown_conditioned_root_log_likelihood(
                    result[0],
                    tree_u_result.root_U[0],
                )
            elif normalize_root_likelihood:
                tree_log_likelihood -= torch.log1p(
                    -tree_u_result.root_U[0, 0]
                )
            LL = LL + tree_log_likelihood
        return LL

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

        tree_log_likelihood = result[0, 0]
        if no_stem:
            tree_log_likelihood = crown_conditioned_root_log_likelihood(
                result[0],
                U_result.root_U[tree_index],
            )
        elif normalize_root_likelihood:
            tree_log_likelihood -= torch.log1p(-U_result.root_U[tree_index, 0])
        LL = LL + tree_log_likelihood

    return LL


def crown_conditioned_root_log_likelihood(
    root_state_log_likelihoods: torch.Tensor,
    root_state_extinction_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Condition a crown tree on both daughter lineages surviving in state 0."""
    if root_state_log_likelihoods.ndim != 1:
        raise ValueError("Root-state log likelihoods must be one-dimensional.")
    if root_state_extinction_probabilities.shape != root_state_log_likelihoods.shape:
        raise ValueError(
            "Root-state extinction probabilities must match root-state likelihoods."
        )
    if root_state_log_likelihoods.numel() == 0:
        raise ValueError("At least one root state is required.")

    return root_state_log_likelihoods[0] - 2.0 * torch.log1p(
        -root_state_extinction_probabilities[0]
    )


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

    if params.per_tree_sampling_rates or params.per_tree_state_sampling_rates:
        logfile.write("sampling_by_tree:\n")
        for tree_index, probabilities in enumerate(
            params.tree_sampling_probabilities.detach().cpu().tolist()
        ):
            values = probabilities if isinstance(probabilities, list) else [probabilities]
            logfile.write(
                f"{tree_index}: "
                + " ".join(f"{value:.15g}" for value in values)
                + "\n"
            )
    elif sampling.ndim == 1:
        logfile.write("sampling: ")
        logfile.write(" ".join(f"{x:.15g}" for x in sampling))
        logfile.write("\n")

    logfile.write("transitions:\n")

    if params.per_tree_transition_rates:
        logfile.write("transition_probabilities_by_tree:\n")
        for tree_index, tree_matrix in enumerate(transitions):
            logfile.write(f"tree_{tree_index}:\n")
            for i in range(params.num_states):
                for j in range(params.num_states):
                    if tree_matrix[i, j] > 0.0:
                        logfile.write(f"{i}->{j} {tree_matrix[i, j]:.15g}\n")
    else:
        for i, j in params.free_edges:
            logfile.write(f"{i}->{j} " f"{transitions[i, j]:.15g}\n")

    logfile.write("transition_times:\n")

    for t in times:

        logfile.write(f"{t:.15g} ")

    logfile.write("\n\n")
    if params.per_tree_transition_rates:
        logfile.write("transition_times_by_tree:\n")
        for tree_index, tree_times in enumerate(params.tree_transition_times):
            values = tree_times.detach().cpu().tolist()
            logfile.write(
                f"tree_{tree_index}: "
                + " ".join(f"{value:.15g}" for value in values)
                + "\n"
            )
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
        no_stem: bool = False,
    ):
        super().__init__()

        self.params = params
        self.trees = trees
        self.datastructures = datastructures
        self.normalize_root_likelihood = normalize_root_likelihood
        self.no_stem = no_stem
        self.static_transition_times = params.transition_times.detach().cpu().tolist()
        self.static_tree_transition_times = (
            [
                times.detach().cpu().tolist()
                for times in params.tree_transition_times
            ]
            if params.tree_transition_times is not None
            else None
        )

    def forward(self):

        return calculate_total_log_likelihood(
            params=self.params,
            trees=self.trees,
            datastructures=self.datastructures,
            static_transition_times=self.static_transition_times,
            normalize_root_likelihood=self.normalize_root_likelihood,
            no_stem=self.no_stem,
            static_tree_transition_times=self.static_tree_transition_times,
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
    no_stem: bool = False,
    shared_sampling_rate: bool = False,
    per_tree_sampling_rates: bool = False,
    per_tree_state_sampling_rates: bool = False,
    per_tree_transition_rates: bool = False,
    tree_transition_times: list[list[float]] | None = None,
    tree_free_edges: list[list[tuple[int, int]]] | None = None,
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
    if no_stem and normalize_root_likelihood:
        raise ValueError(
            "no_stem crown conditioning cannot be combined with "
            "normalize_root_likelihood."
        )

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
        num_trees=len(trees),
        per_tree_sampling_rates=per_tree_sampling_rates,
        per_tree_state_sampling_rates=per_tree_state_sampling_rates,
        per_tree_transition_rates=per_tree_transition_rates,
        tree_transition_times=tree_transition_times,
        tree_free_edges=tree_free_edges,
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
            no_stem=no_stem,
        )

        return OptimizationResult(
            log_likelihood=LL.detach(),
            birth_rates=params.birth_rates.detach(),
            death_rates=params.death_rates.detach(),
            sampling_probabilities=(
                params.sampling_probabilities.detach().mean(dim=0)
                if (
                    params.per_tree_sampling_rates
                    or params.per_tree_state_sampling_rates
                )
                else params.sampling_probabilities.detach()
            ),
            transition_probabilities=(
                params.transition_probabilities.detach().mean(dim=0)
                if params.per_tree_transition_rates
                else params.transition_probabilities.detach()
            ),
            transition_times=(params.transition_times.detach()),
            tree_sampling_probabilities=(
                params.tree_sampling_probabilities.detach()
                if (
                    params.per_tree_sampling_rates
                    or params.per_tree_state_sampling_rates
                )
                else None
            ),
            tree_transition_probabilities=(
                params.transition_probabilities.detach()
                if params.per_tree_transition_rates
                else None
            ),
            tree_transition_times=(
                [times.detach().cpu().tolist() for times in params.tree_transition_times]
                if params.per_tree_transition_rates
                else None
            ),
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
            no_stem=no_stem,
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
                normalize_root_likelihood=normalize_root_likelihood,
                no_stem=no_stem,
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
            no_stem=no_stem,
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
        sampling_probabilities=(
            params.sampling_probabilities.detach().mean(dim=0).clone()
            if (
                params.per_tree_sampling_rates
                or params.per_tree_state_sampling_rates
            )
            else params.sampling_probabilities.detach().clone()
        ),
        transition_probabilities=(
            params.transition_probabilities.detach().mean(dim=0).clone()
            if params.per_tree_transition_rates
            else params.transition_probabilities.detach().clone()
        ),
        transition_times=(params.transition_times.detach().clone()),
        tree_sampling_probabilities=(
            params.tree_sampling_probabilities.detach().clone()
            if (
                params.per_tree_sampling_rates
                or params.per_tree_state_sampling_rates
            )
            else None
        ),
        tree_transition_probabilities=(
            params.transition_probabilities.detach().clone()
            if params.per_tree_transition_rates
            else None
        ),
        tree_transition_times=(
            [times.detach().cpu().tolist() for times in params.tree_transition_times]
            if params.per_tree_transition_rates
            else None
        ),
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
