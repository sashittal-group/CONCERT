# closed_form_coord.py

from __future__ import annotations

from typing import Sequence

import torch

from src.dtos import (
    PreparedTree,
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
    for n in range(1, num_steps):
        dt = transition_times[n] - transition_times[n - 1]
        exponential = torch.exp(-rates * dt)
        denominator = 1.0 - C_values[n - 1] * exponential
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
        .sub(1)
        .clamp(min=0, max=num_steps - 1)
    )
    root_epoch_times = transition_times[epoch_indices]
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
    transition_times,
) -> LikelihoodResult:
    """
    PyTorch version of calculate_log_likelihood_closed().

    Tree topology is stored as NumPy arrays in PreparedTree.
    Only numerical quantities needed for differentiation are
    represented as PyTorch tensors.

    transition_times are fixed during optimization.
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
    # STATIC TREE DATA
    #
    # Keep topology as Python integers/lists. This prevents
    # torch.compile from trying to lower NumPy indexing into
    # tensor operations while keeping it out of autograd.
    # ========================================================

    parent = tree.parent
    edge_idx_to_parent = tree.edge_idx_to_parent
    states = tree.state
    children = tree.children

    # LL TABLE
    #
    # Use a Python list instead of writing into a tensor
    # in-place. Each entry is a differentiable tensor.
    # ========================================================

    num_edges = tree.num_edges

    LL_values = [None for _ in range(num_edges)]

    # ========================================================
    # PROCESS EDGES BOTTOM-UP
    # ========================================================

    for node_id in tree.postorder:

        node_id = int(node_id)

        # ----------------------------------------------------
        # Root has no incoming edge.
        # ----------------------------------------------------

        if parent[node_id] == -1:
            continue

        # ----------------------------------------------------
        # Current edge
        # ----------------------------------------------------

        edge_idx = int(edge_idx_to_parent[node_id])

        u = int(parent[node_id])

        v = node_id

        # ----------------------------------------------------
        # Node times
        #
        # time_t is the only Torch representation of node
        # times.
        # ----------------------------------------------------

        parent_time = tree.time[u]
        child_time = tree.time[v]

        # ====================================================
        # BOUNDARY CONDITION
        # ====================================================

        v_children = children[v]

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        if len(v_children) == 0:

            child_state = int(states[v])

            if child_state < 0:
                raise RuntimeError(f"Leaf node {v} has unknown state.")

            # Do not construct a tensor and assign into it.
            #
            # All states except child_state are impossible.
            #
            # This creates:
            #
            #   [ -inf, ..., log(rho_s), ..., -inf ]
            #

            values = []

            for s in range(num_states):

                if s == child_state:

                    values.append(torch.log(sampling_probabilities[s]))

                else:

                    values.append(
                        torch.full(
                            (),
                            NEG_INF,
                            dtype=dtype,
                            device=device,
                        )
                    )

            boundary_condition = torch.stack(values)

        # ----------------------------------------------------
        # Internal binary node
        # ----------------------------------------------------

        elif len(v_children) == 2:

            left_child = int(v_children[0])

            right_child = int(v_children[1])

            left_edge_idx = int(edge_idx_to_parent[left_child])

            right_edge_idx = int(edge_idx_to_parent[right_child])

            left_LL = LL_values[left_edge_idx]

            right_LL = LL_values[right_edge_idx]

            if left_LL is None:
                raise RuntimeError(
                    f"Likelihood for left child edge "
                    f"{left_edge_idx} is unavailable."
                )

            if right_LL is None:
                raise RuntimeError(
                    f"Likelihood for right child edge "
                    f"{right_edge_idx} is unavailable."
                )

            boundary_condition = torch.log(birth_rates) + left_LL + right_LL

        # ----------------------------------------------------
        # Anything other than binary
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

        closest_n = 0

        # ----------------------------------------------------
        # Transition epochs
        #
        # transition_times are fixed, so these comparisons
        # determine the fixed piece of the likelihood.
        # ----------------------------------------------------

        for n in range(1, num_steps):

            time = transition_times[n]

            prev_transition_time = transition_times[n - 1]

            parent_after = parent_time > time

            child_before = time > child_time

            if parent_after:
                closest_n = n

            if parent_after and child_before:

                # --------------------------------------------
                # Continuous propagation
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
                # State transition
                #
                # No log(0):
                # only transitions with positive probability
                # are included.
                # --------------------------------------------

                # Batch all destination states at once. The first dimension
                # of transition_probabilities is the destination state and
                # the second is the source state.
                log_probabilities = torch.log(
                    transition_probabilities.clamp_min(torch.finfo(dtype).tiny)
                )

                previous_LL = torch.logsumexp(
                    previous_LL[None, :] + log_probabilities,
                    dim=1,
                )

                previous_time = time

        # ====================================================
        # FINAL CONTINUOUS PROPAGATION
        # ====================================================

        final_after_previous = parent_time > previous_time

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
        #
        # Python list assignment is not an in-place operation
        # on a differentiable Tensor.
        # ====================================================

        LL_values[edge_idx] = previous_LL

    # ========================================================
    # STACK LL TABLE
    # ========================================================

    if any(value is None for value in LL_values):
        raise RuntimeError("Not all edge likelihoods were computed.")

    LL = torch.stack(
        LL_values,
        dim=0,
    )

    # ========================================================
    # ROOT LIKELIHOOD
    # ========================================================

    root = tree.root

    root_children = children[root]

    if len(root_children) != 1:
        raise RuntimeError("Expected dummy root with one child.")

    child = int(root_children[0])

    root_edge = int(edge_idx_to_parent[child])

    root_loglikelihood = LL[root_edge, 0]

    return LikelihoodResult(
        root_loglikelihood=root_loglikelihood,
        LL=LL,
    )


def calculate_log_likelihood_levelwise(
    tree: PreparedTree,
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    transition_probabilities: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    Cs: torch.Tensor,
    transition_times,
) -> LikelihoodResult:
    """Calculate likelihoods by batching independent tree levels."""

    device = birth_rates.device
    dtype = birth_rates.dtype
    num_states = birth_rates.shape[0]
    log_birth = torch.log(birth_rates)
    log_sampling = torch.log(sampling_probabilities.clamp_min(torch.finfo(dtype).tiny))
    log_probabilities = torch.log(
        transition_probabilities.clamp_min(torch.finfo(dtype).tiny)
    )
    rates = birth_rates - death_rates

    edge_ll = torch.zeros(
        (tree.num_edges, num_states),
        dtype=dtype,
        device=device,
    )

    def propagate_batch(boundary, parent_time, child_time):
        previous_ll = boundary
        previous_time = child_time

        for n in range(1, len(transition_times)):
            transition_time = transition_times[n]
            previous_transition_time = transition_times[n - 1]
            crosses_epoch = (parent_time > transition_time) & (
                transition_time > child_time
            )

            exp_previous = torch.exp(
                -rates[None, :] * (previous_time[:, None] - previous_transition_time)
            )
            exp_current = torch.exp(
                -rates[None, :] * (transition_time - previous_transition_time)
            )
            flow = -rates[None, :] * (transition_time - previous_time[:, None])
            flow = flow + 2.0 * (
                torch.log(1.0 - Cs[n - 1][None, :] * exp_previous)
                - torch.log(1.0 - Cs[n - 1][None, :] * exp_current)
            )

            propagated = previous_ll + flow
            transitioned = torch.logsumexp(
                propagated[:, None, :] + log_probabilities[None, :, :],
                dim=2,
            )
            mask = crosses_epoch[:, None]
            previous_ll = torch.where(mask, transitioned, previous_ll)
            previous_time = torch.where(
                crosses_epoch,
                torch.full_like(previous_time, transition_time),
                previous_time,
            )

        closest_epoch = torch.sum(
            parent_time[:, None]
            > torch.as_tensor(
                transition_times[1:],
                dtype=dtype,
                device=device,
            ),
            dim=1,
        )
        final_transition_time = torch.as_tensor(
            transition_times,
            dtype=dtype,
            device=device,
        )[closest_epoch]
        final_cs = Cs[closest_epoch]

        exp_previous = torch.exp(
            -rates[None, :] * (previous_time[:, None] - final_transition_time[:, None])
        )
        exp_parent = torch.exp(
            -rates[None, :] * (parent_time[:, None] - final_transition_time[:, None])
        )
        final_flow = -rates[None, :] * (parent_time[:, None] - previous_time[:, None])
        final_flow = final_flow + 2.0 * (
            torch.log(1.0 - final_cs * exp_previous)
            - torch.log(1.0 - final_cs * exp_parent)
        )
        return torch.where(
            (parent_time > previous_time)[:, None],
            previous_ll + final_flow,
            previous_ll,
        )

    for batch in tree.level_batches:

        if "leaf_edge_ids" in batch:
            edge_ids = batch["leaf_edge_ids"]
            states = batch["leaf_states"]
            boundary = torch.full(
                (edge_ids.shape[0], num_states),
                -torch.inf,
                dtype=dtype,
                device=device,
            ).scatter(
                1,
                states[:, None],
                log_sampling[states, None],
            )
            values = propagate_batch(
                boundary,
                batch["leaf_parent_times"],
                batch["leaf_child_times"],
            )
            edge_ll = edge_ll.index_copy(0, edge_ids, values)

        if "internal_edge_ids" in batch:
            edge_ids = batch["internal_edge_ids"]
            child_edge_ids = batch["internal_child_edge_ids"]
            boundary = log_birth + edge_ll[child_edge_ids].sum(dim=1)
            values = propagate_batch(
                boundary,
                batch["internal_parent_times"],
                batch["internal_child_times"],
            )
            edge_ll = edge_ll.index_copy(0, edge_ids, values)

    root_child = tree.children[tree.root][0]
    root_edge = tree.edge_idx_to_parent[root_child]

    return LikelihoodResult(
        root_loglikelihood=edge_ll[root_edge, 0],
        LL=edge_ll,
    )


# ============================================================
# LINEAR PRIOR ON TRANSITION TIMES
# ============================================================


# def calculate_linear_prior_for_transition_times(
#     transition_epochs: Sequence[TransitionEpoch],
#     factor,
# ) -> torch.Tensor:
#     """
#     Calculate:

#         prior = factor * sum_k tau_k

#     The transition times can be torch tensors, so the result
#     remains differentiable with respect to trainable times.
#     """

#     if len(transition_epochs) == 0:

#         return torch.tensor(
#             0.0,
#             dtype=torch.float64,
#         )

#     # Determine device/dtype from the first epoch if possible.
#     first_time = transition_epochs[0].time

#     if isinstance(first_time, torch.Tensor):

#         prior = torch.zeros(
#             (),
#             device=first_time.device,
#             dtype=first_time.dtype,
#         )

#     else:

#         prior = torch.tensor(
#             0.0,
#             dtype=torch.float64,
#         )

#     factor_tensor = torch.as_tensor(
#         factor,
#         device=prior.device,
#         dtype=prior.dtype,
#     )

#     for epoch in transition_epochs:

#         time = _epoch_time(
#             epoch,
#             prior.device,
#             prior.dtype,
#         )

#         prior = prior + factor_tensor * time

#     return prior
