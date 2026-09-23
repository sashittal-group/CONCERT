from dataclasses import dataclass
from typing import Optional

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
    GPU/autograd-friendly tree representation.

    N = number of nodes
    E = number of edges
    """

    root: int

    # Node attributes (shape N)
    original_id: torch.Tensor          # int64
    time: torch.Tensor                 # float64
    state: torch.Tensor                # int64
    parent: torch.Tensor               # int64
    edge_idx_to_parent: torch.Tensor   # int64

    # Edge attributes (shape E)
    edge_parent: torch.Tensor          # int64
    edge_child: torch.Tensor           # int64

    # Traversal order (shape N)
    postorder: torch.Tensor            # int64

    @property
    def num_nodes(self):
        return self.time.shape[0]

    @property
    def num_edges(self):
        return self.edge_parent.shape[0]


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
# DIFFERENTIATION TREE
# ============================================================

@dataclass
class DiffTree:
    """
    Replacement for Diff_T_Node pointer tree.

    parent[root] = -1
    """

    state: torch.Tensor       # int64
    diff_time: torch.Tensor   # float64
    parent: torch.Tensor      # int64

    @property
    def num_nodes(self):
        return self.state.shape[0]


# ============================================================
# COMPUTE GRAPH STRUCTURES
# ============================================================

@dataclass
class ComputeEdge:
    id: int

    u: int
    v: int

    u_T: float
    v_T: float

    v_state: int


@dataclass
class ComputeNode:
    id: int
    T: float
    is_leaf: bool
    state: int


# ============================================================
# TRANSITION EPOCHS
# ============================================================

@dataclass
class TransitionEpoch:
    """
    transition_edges shape:
        (k, 2)

    each row:
        [parent_state, child_state]
    """

    time: float
    transition_edges: list[tuple[int, int]]


@dataclass
class TransitionEpochWithEdge:
    """
    edge shape:
        (2,)
        [u, v]
    """

    time: float
    transition_edges: torch.Tensor
    edge: torch.Tensor


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


# ============================================================
# OPTIMIZATION OUTPUT
# ============================================================

@dataclass
class OptimizationResult:
    log_likelihood: float

    birth_rates: torch.Tensor
    death_rates: torch.Tensor

    sampling_probabilities: torch.Tensor

    transition_probabilities: torch.Tensor

    transition_times: torch.Tensor


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
