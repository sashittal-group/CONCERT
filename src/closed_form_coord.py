# closed_form_coord.py

from __future__ import annotations

from typing import Sequence

import torch

from src.dtos import (
    PreparedTree,
    TransitionEpoch,
    LikelihoodResult,
    UResult,
)


# ============================================================
# CONSTANTS
# ============================================================

NEG_INF = -torch.inf


# ============================================================
# HELPERS
# ============================================================


def _epoch_time(
    epoch: TransitionEpoch,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return epoch.time as a scalar torch tensor.

    If epoch.time is already a tensor depending on trainable
    parameters, the autograd graph is preserved.
    """

    if isinstance(epoch.time, torch.Tensor):
        return epoch.time.to(
            device=device,
            dtype=dtype,
        )

    return torch.tensor(
        epoch.time,
        device=device,
        dtype=dtype,
    )


def _get_tree_topology(tree: PreparedTree):
    """
    Build and cache the static tree topology.

    This replaces the C++ vector<vector<int>> children
    structure without requiring Python object pointers.

    The topology is independent of the model parameters, so
    it is computed only once per PreparedTree.

    Cached fields:

        children[node]      -> list of child node IDs
        parent[node]        -> parent node ID
        edge_idx[node]      -> edge index from parent to node
        state[node]         -> observed state

    All of these are Python/integer structures because they are
    static and do not participate in autograd.
    """

    cache = getattr(
        tree,
        "_topology_cache",
        None,
    )

    if cache is not None:
        return cache

    # Move only the static topology to CPU once.
    parent = tree.parent.detach().cpu().tolist()

    edge_idx_to_parent = tree.edge_idx_to_parent.detach().cpu().tolist()

    states = tree.state.detach().cpu().tolist()

    edge_parent = tree.edge_parent.detach().cpu().tolist()

    edge_child = tree.edge_child.detach().cpu().tolist()

    children = [[] for _ in range(tree.num_nodes)]

    for u, v in zip(
        edge_parent,
        edge_child,
    ):
        children[u].append(v)

    cache = {
        "children": children,
        "parent": parent,
        "edge_idx_to_parent": edge_idx_to_parent,
        "state": states,
    }

    # Cache directly on the PreparedTree.
    tree._topology_cache = cache

    return cache


# ============================================================
# CALCULATE U, FLOW_U, C
# ============================================================


def calculate_Us(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    transition_probabilities: torch.Tensor,
    transition_times: torch.Tensor,
) -> UResult:

    if birth_rates.ndim != 1:
        raise ValueError("birth_rates must have shape (num_states,)")

    if death_rates.shape != birth_rates.shape:
        raise ValueError("death_rates must have the same shape as " "birth_rates")

    if sampling_probabilities.shape != birth_rates.shape:
        raise ValueError("sampling_probabilities must have shape " "(num_states,)")

    num_states = birth_rates.shape[0]
    num_steps = len(transition_times)

    if num_steps == 0:
        raise ValueError("transition_times cannot be empty")

    # --------------------------------------------------------
    # U[0,:] = 1 - rho
    #
    # C[0,:] =
    #
    #     (U[0] - mu/lambda)
    #     ------------------
    #         U[0] - 1
    # --------------------------------------------------------

    U0 = 1.0 - sampling_probabilities

    numerator = U0 - death_rates / birth_rates

    denominator = U0 - 1.0

    C0 = numerator / denominator

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
    # Epoch recursion
    # --------------------------------------------------------

    for n in range(1, num_steps):

        prev_time = transition_times[n - 1]
        time = transition_times[n]

        dt = time - prev_time

        r = birth_rates - death_rates

        # ----------------------------------------------------
        # Continuous propagation
        #
        # flow_U[n,s] =
        #
        # 1 -
        # (lambda_s - mu_s)
        # --------------------------
        # lambda_s *
        # [1 - C[n-1,s] exp(-r_s dt)]
        # ----------------------------------------------------

        exponential = torch.exp(-r * dt)

        denominator = 1.0 - C_values[n - 1] * exponential

        flow_U_n = 1.0 - r / (birth_rates * denominator)

        # ----------------------------------------------------
        # State transition
        #
        # U[n,s] =
        #     sum_j Q[s,j] flow_U[n,j]
        #
        # U[n] = Q @ flow_U[n]
        # ----------------------------------------------------

        U_n = transition_probabilities @ flow_U_n

        # ----------------------------------------------------
        # C[n]
        # ----------------------------------------------------

        numerator = U_n - death_rates / birth_rates

        denominator = U_n - 1.0

        C_n = numerator / denominator

        # ----------------------------------------------------
        # Append new tensors.
        # ----------------------------------------------------

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

    return UResult(
        U=U,
        flow_U=flow_Us,
        Cs=Cs,
    )


# ============================================================
# CALCULATE TREE LOG LIKELIHOOD
# ============================================================


def calculate_log_likelihood_closed(
    tree: PreparedTree,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    transition_probabilities: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    Cs: torch.Tensor,
    transition_times: torch.Tensor,
) -> LikelihoodResult:
    """
    PyTorch version of calculate_log_likelihood_closed().

    The computation follows the C++ implementation:

        1. Allocate LL[edge, state].
        2. Traverse the tree bottom-up.
        3. Construct boundary condition for each node.
        4. Propagate continuously through each relevant epoch.
        5. Apply state-transition probabilities at epochs.
        6. Continue from the final relevant epoch to the parent.
        7. Return LL at the dummy root edge, state 0.
    """

    num_states = birth_rates.shape[0]
    num_steps = len(transition_times)

    if num_steps == 0:
        raise ValueError("transition_times cannot be empty")

    if death_rates.shape != birth_rates.shape:
        raise ValueError("death_rates must have same shape as birth_rates")

    if sampling_probabilities.shape != birth_rates.shape:
        raise ValueError("sampling_probabilities must have shape " "(num_states,)")

    if transition_probabilities.shape != (
        num_states,
        num_states,
    ):
        raise ValueError(
            "transition_probabilities must have shape " "(num_states, num_states)"
        )

    if Cs.shape != (
        num_steps,
        num_states,
    ):
        raise ValueError("Cs must have shape " "(num_steps, num_states)")

    device = birth_rates.device
    dtype = birth_rates.dtype

    # ========================================================
    # STATIC TREE TOPOLOGY
    # ========================================================

    topology = _get_tree_topology(tree)

    children = topology["children"]
    parent = topology["parent"]
    edge_idx_to_parent = topology["edge_idx_to_parent"]
    states = topology["state"]

    # ========================================================
    # LL TABLE
    #
    # LL[e,s] = likelihood of edge e conditional on
    # parent being state s.
    # ========================================================

    num_edges = tree.num_edges

    LL = torch.full(
        (num_edges, num_states),
        NEG_INF,
        device=device,
        dtype=dtype,
    )

    # ========================================================
    # PROCESS EDGES BOTTOM-UP
    # ========================================================

    for node_id_tensor in tree.postorder:

        # postorder is an int64 tensor. Convert only the
        # structural index to Python int.
        node_id = int(node_id_tensor.detach().item())

        # Root has no parent and is not represented by an
        # incoming likelihood edge.
        if parent[node_id] == -1:
            continue

        # ----------------------------------------------------
        # Current edge
        # ----------------------------------------------------

        edge_idx = edge_idx_to_parent[node_id]

        u = parent[node_id]
        v = node_id

        # Node times remain tensors on the selected device.
        parent_time = tree.time[u]
        child_time = tree.time[v]

        # ====================================================
        # BOUNDARY CONDITION
        # ====================================================

        boundary_condition = torch.full(
            (num_states,),
            NEG_INF,
            device=device,
            dtype=dtype,
        )

        v_children = children[v]

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        if len(v_children) == 0:

            child_state = states[v]

            if child_state < 0:
                raise ValueError(f"Leaf node {v} has unknown state.")

            boundary_condition[child_state] = torch.log(
                sampling_probabilities[child_state]
            )

        # ----------------------------------------------------
        # Internal binary node
        # ----------------------------------------------------

        elif len(v_children) == 2:

            left_child = v_children[0]
            right_child = v_children[1]

            left_edge_idx = edge_idx_to_parent[left_child]

            right_edge_idx = edge_idx_to_parent[right_child]

            boundary_condition = (
                torch.log(birth_rates) + LL[left_edge_idx] + LL[right_edge_idx]
            )

        # ----------------------------------------------------
        # Anything other than 0 or 2 children is invalid.
        # ----------------------------------------------------

        else:

            raise RuntimeError(
                f"Expected 0 or 2 children, but node "
                f"{v} has {len(v_children)} children."
            )

        # ====================================================
        # INTEGRATE UPWARD
        # ====================================================

        previous_LL = boundary_condition

        previous_time = child_time

        # Closest transition epoch before parent_time.
        #
        # The value is a discrete index used only for deciding
        # which continuous formula applies. The actual times
        # used in the likelihood remain tensors.
        closest_n = 0

        for n in range(1, num_steps):

            time = transition_times[n]
            prev_transition_time = transition_times[n - 1]

            # ------------------------------------------------
            # Determine whether this transition lies between
            # child and parent.
            #
            # This is discrete control flow. Gradients through
            # the actual time values inside the selected branch
            # are still preserved.
            # ------------------------------------------------

            parent_after = bool((parent_time > time).detach().item())

            child_before = bool((time > child_time).detach().item())

            if parent_after:
                closest_n = n

            if parent_after and child_before:

                # --------------------------------------------
                # Continuous propagation from previous_time
                # to time
                # --------------------------------------------

                r = birth_rates - death_rates

                delta = time - previous_time

                exp_previous = torch.exp(-r * (previous_time - prev_transition_time))

                exp_current = torch.exp(-r * (time - prev_transition_time))

                log_previous = torch.log(1.0 - Cs[n - 1] * exp_previous)

                log_current = torch.log(1.0 - Cs[n - 1] * exp_current)

                flow = -r * delta + 2.0 * (log_previous - log_current)

                previous_LL = previous_LL + flow

                # --------------------------------------------
                # Apply state transition
                #
                # current_LL[s] =
                #
                # log sum_j(
                #     exp(previous_LL[j])
                #     * Q[s,j]
                # )
                #
                # = logsumexp(
                #       previous_LL[j] + log Q[s,j]
                #
                # This replaces the C++ temporary 'terms'
                # vector and is directly GPU-compatible.
                # --------------------------------------------

                # --------------------------------------------
                # Apply state transition
                # --------------------------------------------

                current_LL_values = []

                for s in range(num_states):

                    # --------------------------------------------------------
                    # Find the transitions that actually exist in this row.
                    # --------------------------------------------------------

                    nonzero = transition_probabilities[s] > 0

                    # --------------------------------------------------------
                    # Only self-transition exists.
                    #
                    # Therefore P[s,s] = 1 exactly.
                    # --------------------------------------------------------

                    if nonzero.sum() == 1:

                        current_LL_values.append(previous_LL[s])

                    else:

                        # ----------------------------------------------------
                        # Extract only positive transition probabilities.
                        #
                        # Therefore torch.log() never sees zero.
                        # ----------------------------------------------------

                        probabilities = transition_probabilities[s][nonzero]

                        log_probabilities = torch.log(probabilities)

                        likelihood_terms = previous_LL[nonzero] + log_probabilities

                        current_LL_values.append(
                            torch.logsumexp(
                                likelihood_terms,
                                dim=0,
                            )
                        )

                previous_LL = torch.stack(current_LL_values)
                previous_time = time

        # ====================================================
        # FINAL CONTINUOUS PROPAGATION
        # ====================================================

        final_after_previous = bool((parent_time > previous_time).detach().item())

        if final_after_previous:

            n = closest_n

            transition_time_n = transition_times[n]

            r = birth_rates - death_rates

            delta = parent_time - previous_time

            exp_previous = torch.exp(-r * (previous_time - transition_time_n))

            exp_parent = torch.exp(-r * (parent_time - transition_time_n))

            log_previous = torch.log(1.0 - Cs[n] * exp_previous)

            log_parent = torch.log(1.0 - Cs[n] * exp_parent)

            flow = -r * delta + 2.0 * (log_previous - log_parent)

            previous_LL = previous_LL + flow

        # ====================================================
        # STORE EDGE LIKELIHOOD
        # ====================================================

        LL[edge_idx] = previous_LL

    # ========================================================
    # ROOT LIKELIHOOD
    # ========================================================

    root = tree.root

    root_children = children[root]

    if len(root_children) != 1:

        raise RuntimeError("Expected dummy root with one child.")

    child = root_children[0]

    root_edge = edge_idx_to_parent[child]

    root_loglikelihood = LL[root_edge, 0]

    return LikelihoodResult(
        root_loglikelihood=root_loglikelihood,
        LL=LL,
    )


# ============================================================
# LINEAR PRIOR ON TRANSITION TIMES
# ============================================================


def calculate_linear_prior_for_transition_times(
    transition_epochs: Sequence[TransitionEpoch],
    factor,
) -> torch.Tensor:
    """
    Calculate:

        prior = factor * sum_k tau_k

    The transition times can be torch tensors, so the result
    remains differentiable with respect to trainable times.
    """

    if len(transition_epochs) == 0:

        return torch.tensor(
            0.0,
            dtype=torch.float64,
        )

    # Determine device/dtype from the first epoch if possible.
    first_time = transition_epochs[0].time

    if isinstance(first_time, torch.Tensor):

        prior = torch.zeros(
            (),
            device=first_time.device,
            dtype=first_time.dtype,
        )

    else:

        prior = torch.tensor(
            0.0,
            dtype=torch.float64,
        )

    factor_tensor = torch.as_tensor(
        factor,
        device=prior.device,
        dtype=prior.dtype,
    )

    for epoch in transition_epochs:

        time = _epoch_time(
            epoch,
            prior.device,
            prior.dtype,
        )

        prior = prior + factor_tensor * time

    return prior
