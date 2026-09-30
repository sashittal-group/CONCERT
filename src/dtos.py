from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ============================================================
# RAW INPUT STRUCTURES
# ============================================================

@dataclass
class Node:
    id: int
    T: float
    is_leaf: bool
    state: int  # -1 if unknown


@dataclass
class Edge:
    u: int
    v: int
    length: float


# ============================================================
# PREPARED TREE
# ============================================================

@dataclass
class PreparedTree:
    """
    Static tree representation.

    Static tree data are stored as Python lists so they can be
    embedded in a compiled graph without NumPy indexing.
    `time_t` is the Torch version of `time` used by the
    differentiable likelihood.
    """

    # ========================================================
    # ROOT
    # ========================================================

    root: int

    # ========================================================
    # NODE ATTRIBUTES
    # ========================================================

    original_id: list[int]
    time: list[float]
    state: list[int]
    parent: list[int]
    children: list[list[int]]
    edge_idx_to_parent: list[int]

    # ========================================================
    # EDGE ATTRIBUTES
    # ========================================================

    edge_parent: list[int]
    edge_child: list[int]

    # ========================================================
    # POSTORDER
    # ========================================================

    postorder: list[int]
    levels: list[list[int]]
    level_batches: list[dict[str, torch.Tensor]]

    # ========================================================
    # TORCH TIME
    # ========================================================

    time_t: torch.Tensor

    @property
    def num_nodes(self):
        return len(self.time)

    @property
    def num_edges(self):
        return len(self.edge_parent)


# ============================================================
# LIKELIHOOD OUTPUT
# ============================================================

@dataclass
class LikelihoodResult:
    """
    LL shape:
        (num_nodes, num_states)
    """

    root_loglikelihood: torch.Tensor
    LL: torch.Tensor



# ============================================================
# U CALCULATIONS
# ============================================================

@dataclass
class UResult:
    """
    Typical shapes:

    U:
        (num_epochs, num_states)

    flow_U:
        (num_epochs, num_states)

    Cs:
        (num_epochs, num_states)
    """

    U: torch.Tensor
    flow_U: torch.Tensor
    Cs: torch.Tensor
    root_U: torch.Tensor


# ============================================================
# OPTIMIZATION OUTPUT
# ============================================================

@dataclass
class OptimizationResult:
    log_likelihood: float

    birth_rates: torch.Tensor
    death_rates: torch.Tensor

    sampling_probabilities: torch.Tensor


# ============================================================
# DEVICE HELPERS
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

torch.set_default_dtype(torch.float64)
