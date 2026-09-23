from collections import defaultdict
import csv
import random
import string
import sys
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import numpy as np
import pandas as pd
import argparse
import os


def simulate_constant_bd(lam, mu, time_length, seed=None):
    if seed is not None:
        random.seed(seed)

    G = nx.DiGraph()

    # node ids
    next_node = 0
    root = next_node
    next_node += 1

    G.add_node(root, time=0.0, event="root")

    alive = []

    child = next_node
    next_node += 1
    G.add_node(child, time=None, event=None)
    G.add_edge(root, child, start_time=0.0, end_time=None, length=None)
    alive.append((root, child))

    t = 0.0

    while t < time_length and alive:
        k = len(alive)
        R = k * (lam + mu)
        if R == 0:
            break

        dt = random.expovariate(R)

        if t + dt >= time_length:
            break

        t += dt

        i = random.randrange(len(alive))
        parent, tip = alive.pop(i)

        if random.random() < lam / (lam + mu):
            # ---------- birth ----------
            G.nodes[tip]["time"] = t
            G.nodes[tip]["event"] = "birth"

            G.edges[parent, tip]["end_time"] = t
            G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

            for _ in range(2):
                c = next_node
                next_node += 1
                G.add_node(c, time=None, event=None)
                G.add_edge(
                    tip,
                    c,
                    start_time=t,
                    end_time=None,
                    length=None
                )
                alive.append((tip, c))

        else:
            # ---------- death ----------
            G.nodes[tip]["time"] = t
            G.nodes[tip]["event"] = "death"

            G.edges[parent, tip]["end_time"] = t
            G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

    # truncate surviving lineages at T
    for parent, tip in alive:
        G.nodes[tip]["time"] = time_length
        G.nodes[tip]["event"] = "sample"

        G.edges[parent, tip]["end_time"] = time_length
        G.edges[parent, tip]["length"] = time_length - G.edges[parent, tip]["start_time"]

    return G, root


def simulate_multitype_bd_with_diff(
    birth_rates,
    death_rates,
    sampling_rates,
    transition_probabilities,
    differentiation_times,
    time_length,
    root_state="A",
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.DiGraph()
    next_node = 0

    root = next_node
    next_node += 1
    G.add_node(root, time=0.0, event="root", state=root_state)

    child = next_node
    next_node += 1
    G.add_node(child, time=None, event=None, state=root_state)
    G.add_edge(root, child, start_time=0.0, end_time=None, length=None)

    alive = [(root, child, root_state)]

    t = 0.0

    while t < time_length and alive:

        # ---- compute BD total rate ----
        rates = []
        total_rate = 0.0

        for parent, tip, state in alive:
            r = birth_rates[state] + death_rates[state]
            rates.append(r)
            total_rate += r

        # BD event time
        if total_rate > 0:
            dt_bd = random.expovariate(total_rate)
            t_bd = t + dt_bd
        else:
            t_bd = float("inf")

        # ---- compute next differentiation time ----
        t_transition = float("inf")

        for i, (parent, tip, state) in enumerate(alive):
            transition_times = differentiation_times.nodes[state]["transition_times"]
            for transition_time in transition_times:
                if transition_time > t and transition_time < t_transition:
                    t_transition = transition_time
            
        # ---- determine next event ----
        if t_bd < t_transition and t_bd < time_length:
            # ===== Birth or Death =====
            t = t_bd

            r = random.random() * total_rate
            cumulative = 0
            for i, rate in enumerate(rates):
                cumulative += rate
                if r <= cumulative:
                    chosen_index = i
                    break

            parent, tip, state = alive.pop(chosen_index)

            G.nodes[tip]["time"] = t
            G.edges[parent, tip]["end_time"] = t
            G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

            if random.random() < birth_rates[state] / (birth_rates[state] + death_rates[state]):
                # birth
                G.nodes[tip]["event"] = "birth"

                for _ in range(2):
                    c = next_node
                    next_node += 1
                    G.add_node(c, time=None, event=None, state=state)
                    G.add_edge(tip, c, start_time=t, end_time=None, length=None)
                    alive.append((tip, c, state))
            else:
                # death
                G.nodes[tip]["event"] = "death"

        elif t_transition < time_length:
            # ===== Differentiation =====
            t = t_transition

            new_alive = []

            for parent, tip, state in alive:
                # close current edge
                G.nodes[tip]["time"] = t
                G.nodes[tip]["event"] = "transition"

                G.edges[parent, tip]["end_time"] = t
                G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

                next_states = list(differentiation_times.successors(state))  + [state]

                # Extract probabilities for allowed children
                probs = (
                    transition_probabilities.loc[state, next_states]
                    .astype(float)
                    .tolist()
                )

                # Optional safety normalization (recommended)
                total = np.sum(probs)
                if total <= 0:
                    raise ValueError(f"No positive transition probability from {state}")
                probs = np.array(probs) / total

                # Sample according to probability vector
                new_state = np.random.choice(next_states, p=probs)

                c = next_node
                next_node += 1
                G.add_node(c, time=None, event=None, state=new_state)
                G.add_edge(tip, c, start_time=t, end_time=None, length=None)

                new_alive.append((tip, c, new_state))

            alive = new_alive

        else:
            break

    # ---- truncate survivors ----

    for parent, tip, state in alive:

        edge = G.edges[parent, tip]
        edge["end_time"] = time_length
        edge["length"] = time_length - edge["start_time"]
        edge["end_state"] = state
        G.nodes[tip]["time"] = time_length

        if np.random.random() < sampling_rates[state]:
            G.nodes[tip]["event"] = "sample"
        else:
            G.nodes[tip]["event"] = "unsampled"
    
    return G, root


def simulate_multitype_bd_ctmc(
    birth_rates,
    death_rates,
    sampling_rates,
    Q,
    T,
    root_state="A",
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.DiGraph()

    next_node = 0

    root = next_node
    next_node += 1

    G.add_node(
        root,
        time=0.0,
        event="root",
        state=root_state
    )

    child = next_node
    next_node += 1

    G.add_node(
        child,
        time=None,
        event=None,
        state=root_state
    )

    G.add_edge(
        root,
        child,
        start_time=0.0,
        end_time=None,
        length=None
    )

    alive = [(root, child, root_state)]

    t = 0.0

    while alive and t < T:

        lineage_rates = []
        total_rate = 0.0

        for parent, tip, state in alive:

            q_out = float(Q.loc[state].sum())

            r = (
                birth_rates[state]
                + death_rates[state]
                + q_out
            )

            lineage_rates.append(r)
            total_rate += r

        if total_rate == 0:
            break

        dt = np.random.exponential(
            1.0 / total_rate
        )

        if t + dt >= T:
            break

        t += dt

        # choose lineage

        r = np.random.uniform(0, total_rate)

        cumulative = 0.0

        for idx, rate in enumerate(lineage_rates):
            cumulative += rate

            if r <= cumulative:
                chosen = idx
                break

        parent, tip, state = alive.pop(chosen)

        lam = birth_rates[state]
        mu = death_rates[state]

        transitions = {
            d: float(Q.loc[state, d])
            for d in Q.columns
            if float(Q.loc[state, d]) > 0
        }

        event_rates = [lam, mu]
        event_labels = ["birth", "death"]

        for d, q in transitions.items():
            event_rates.append(q)
            event_labels.append(("transition", d))

        total = sum(event_rates)

        u = np.random.uniform(0, total)

        cumulative = 0.0

        chosen_event = None

        for rate, label in zip(
            event_rates,
            event_labels
        ):
            cumulative += rate

            if u <= cumulative:
                chosen_event = label
                break

        # close current branch

        G.nodes[tip]["time"] = t
        G.edges[parent, tip]["end_time"] = t
        G.edges[parent, tip]["length"] = (t - G.edges[parent, tip]["start_time"])

        if chosen_event == "birth":

            G.nodes[tip]["event"] = "birth"

            for _ in range(2):

                c = next_node
                next_node += 1

                G.add_node(
                    c,
                    time=None,
                    event=None,
                    state=state
                )

                G.add_edge(
                    tip,
                    c,
                    start_time=t,
                    end_time=None,
                    length=None
                )

                alive.append(
                    (tip, c, state)
                )

        elif chosen_event == "death":

            G.nodes[tip]["event"] = "death"

        else:

            _, new_state = chosen_event

            G.nodes[tip]["event"] = "transition"

            c = next_node
            next_node += 1

            G.add_node(
                c,
                time=None,
                event=None,
                state=new_state
            )

            G.add_edge(
                tip,
                c,
                start_time=t,
                end_time=None,
                length=None
            )

            alive.append(
                (tip, c, new_state)
            )

    # truncate surviving branches

    for parent, tip, state in alive:

        G.nodes[tip]["time"] = T

        G.edges[parent, tip]["end_time"] = T
        G.edges[parent, tip]["length"] = (
            T - G.edges[parent, tip]["start_time"]
        )

        if np.random.random() < sampling_rates[state]:
            G.nodes[tip]["event"] = "sample"
        else:
            G.nodes[tip]["event"] = "unsampled"


    return G, root


def simulate_multitype_bd_ctmc2(
    birth_rates,
    death_rates,
    sampling_rates,
    Q,
    T,
    root_state="A",
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.DiGraph()

    next_node = 0

    # root
    root = next_node
    next_node += 1

    G.add_node(
        root,
        time=0.0,
        event="root",
        state=root_state
    )

    # first lineage
    child = next_node
    next_node += 1

    G.add_node(
        child,
        time=None,
        event=None,
        state=root_state
    )

    G.add_edge(
        root,
        child,
        start_time=0.0,
        end_time=None,
        length=None,
        start_state=root_state,
        end_state=root_state,
        transitions=[]
    )

    # alive lineages:
    # (parent node, current node, current state)
    alive = [
        (root, child, root_state)
    ]

    t = 0.0


    while alive and t < T:

        lineage_rates = []
        total_rate = 0.0


        # calculate total event rate
        for parent, tip, state in alive:

            q_out = float(Q.loc[state].sum())

            r = (
                birth_rates[state]
                + death_rates[state]
                + q_out
            )

            lineage_rates.append(r)
            total_rate += r


        if total_rate == 0:
            break


        dt = np.random.exponential(
            1.0 / total_rate
        )


        if t + dt >= T:
            break


        t += dt


        # choose lineage experiencing event

        r = np.random.uniform(
            0,
            total_rate
        )

        cumulative = 0.0

        chosen = None

        for idx, rate in enumerate(lineage_rates):

            cumulative += rate

            if r <= cumulative:
                chosen = idx
                break


        parent, tip, state = alive.pop(chosen)


        # choose event on lineage

        lam = birth_rates[state]
        mu = death_rates[state]


        transitions = {
            d: float(Q.loc[state, d])
            for d in Q.columns
            if float(Q.loc[state, d]) > 0
        }


        event_rates = [
            lam,
            mu
        ]

        event_labels = [
            "birth",
            "death"
        ]


        for d, q in transitions.items():

            event_rates.append(q)

            event_labels.append(
                ("transition", d)
            )


        total = sum(event_rates)


        u = np.random.uniform(
            0,
            total
        )

        cumulative = 0.0

        chosen_event = None


        for rate, label in zip(
            event_rates,
            event_labels
        ):

            cumulative += rate

            if u <= cumulative:
                chosen_event = label
                break



        # ----------------------------------------------------
        # close branch if birth/death
        # transition does NOT close branch
        # ----------------------------------------------------

        if chosen_event == "birth":

            edge = G.edges[parent, tip]

            edge["end_time"] = t
            edge["length"] = (
                t - edge["start_time"]
            )
            edge["end_state"] = state


            G.nodes[tip]["time"] = t
            G.nodes[tip]["event"] = "birth"


            # create two descendant lineages

            for _ in range(2):

                c = next_node
                next_node += 1


                G.add_node(
                    c,
                    time=None,
                    event=None,
                    state=state
                )


                G.add_edge(
                    tip,
                    c,
                    start_time=t,
                    end_time=None,
                    length=None,
                    start_state=state,
                    end_state=state,
                    transitions=[]
                )


                alive.append(
                    (
                        tip,
                        c,
                        state
                    )
                )



        elif chosen_event == "death":

            edge = G.edges[parent, tip]

            edge["end_time"] = t
            edge["length"] = (
                t - edge["start_time"]
            )
            edge["end_state"] = state


            G.nodes[tip]["time"] = t
            G.nodes[tip]["event"] = "death"



        else:

            # ------------------------------
            # CTMC state transition
            # keep same edge
            # ------------------------------

            _, new_state = chosen_event


            edge = G.edges[parent, tip]


            edge["transitions"].append(
                {
                    "time": t,
                    "from": state,
                    "to": new_state
                }
            )


            edge["end_state"] = new_state


            # update current state
            alive.append(
                (
                    parent,
                    tip,
                    new_state
                )
            )



    # ----------------------------------------------------
    # truncate remaining lineages at T
    # ----------------------------------------------------

    for parent, tip, state in alive:

        edge = G.edges[parent, tip]


        edge["end_time"] = T

        edge["length"] = (
            T - edge["start_time"]
        )

        edge["end_state"] = state


        G.nodes[tip]["time"] = T


        if np.random.random() < sampling_rates[state]:

            G.nodes[tip]["event"] = "sample"

        else:

            G.nodes[tip]["event"] = "unsampled"



    return G, root

import random
import numpy as np
import networkx as nx


def get_active_transitions(
    state,
    t,
    Q,
    transition_windows,
):
    """
    Returns a dictionary

        {destination_state: rate}

    containing only transitions that are active at time t.

    transition_windows format:

    {
        ("A", "B"): [(0.2, 0.4), (0.8, 1.0)],
        ("A", "C"): [(0.5, 0.6)],
        ("B", "D"): [(0.3, 0.9)],
    }
    """

    transitions = {}

    for dest in Q.columns:

        q = float(Q.loc[state, dest])

        if q <= 0:
            continue

        active = False

        for start, end in transition_windows.get(
            (state, dest),
            []
        ):
            if start <= t < end:
                active = True
                break

        if active:
            transitions[dest] = q

    return transitions


def next_rate_change(
    t,
    transition_windows,
):
    """
    Returns the next transition-window boundary
    strictly greater than t.

    Returns None if no future boundary exists.
    """

    boundary = None

    for windows in transition_windows.values():

        for start, end in windows:

            if start > t:
                boundary = (
                    start
                    if boundary is None
                    else min(boundary, start)
                )

            if end > t:
                boundary = (
                    end
                    if boundary is None
                    else min(boundary, end)
                )

    return boundary


def close_branch(
    G,
    parent,
    tip,
    t,
):
    """
    Close an active branch at time t.
    """

    G.nodes[tip]["time"] = t
    G.edges[parent, tip]["end_time"] = t
    G.edges[parent, tip]["length"] = (
        t - G.edges[parent, tip]["start_time"]
    )


def simulate_multitype_bdt_ctmc_window(
    birth_rates,
    death_rates,
    sampling_rates,
    Q,
    transition_windows,
    T,
    root_state="A",
    seed=None,
):
    """
    Exact Gillespie simulation of a multitype birth-death process
    with transition-specific activation windows.

    Parameters
    ----------
    birth_rates : dict
        state -> lambda

    death_rates : dict
        state -> mu

    sampling_rates : dict
        state -> rho

    Q : pandas.DataFrame
        transition-rate matrix

    transition_windows : dict

        {
            ("A","B"): [(0.2,0.4), (0.8,1.0)],
            ("A","C"): [(0.5,0.6)],
            ...
        }

    T : float
        final time
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.DiGraph()

    next_node = 0

    root = next_node
    next_node += 1

    G.add_node(
        root,
        time=0.0,
        event="root",
        state=root_state,
    )

    first = next_node
    next_node += 1

    G.add_node(
        first,
        time=None,
        event=None,
        state=root_state,
    )

    G.add_edge(
        root,
        first,
        start_time=0.0,
        end_time=None,
        length=None,
    )

    alive = [
        (root, first, root_state)
    ]

    t = 0.0

    while alive and t < T:

        #
        # Build complete event list
        #
        events = []

        for lineage_idx, (
            parent,
            tip,
            state,
        ) in enumerate(alive):

            lam = float(
                birth_rates[state]
            )

            mu = float(
                death_rates[state]
            )

            if lam > 0:
                events.append(
                    (
                        lineage_idx,
                        "birth",
                        lam,
                    )
                )

            if mu > 0:
                events.append(
                    (
                        lineage_idx,
                        "death",
                        mu,
                    )
                )

            transitions = get_active_transitions(
                state,
                t,
                Q,
                transition_windows,
            )

            for dest, q in transitions.items():

                events.append(
                    (
                        lineage_idx,
                        ("transition", dest),
                        q,
                    )
                )

        total_rate = sum(
            rate
            for _, _, rate in events
        )

        if total_rate <= 0:
            break

        #
        # Draw waiting time
        #
        dt = np.random.exponential(
            1.0 / total_rate
        )

        boundary = next_rate_change(
            t,
            transition_windows,
        )

        #
        # If rate structure changes first,
        # move to boundary and restart.
        #
        if (
            boundary is not None
            and t + dt >= boundary
        ):
            t = boundary
            continue

        #
        # Event would happen after T
        #
        if t + dt >= T:
            break

        t += dt

        #
        # Choose event
        #
        r = np.random.uniform(
            0.0,
            total_rate,
        )

        cumulative = 0.0

        chosen_event = None

        for event in events:

            cumulative += event[2]

            if r <= cumulative:
                chosen_event = event
                break

        lineage_idx, event_type, _ = chosen_event

        parent, tip, state = alive.pop(
            lineage_idx
        )

        close_branch(
            G,
            parent,
            tip,
            t,
        )

        #
        # Birth
        #
        if event_type == "birth":

            G.nodes[tip]["event"] = "birth"

            for _ in range(2):

                child = next_node
                next_node += 1

                G.add_node(
                    child,
                    time=None,
                    event=None,
                    state=state,
                )

                G.add_edge(
                    tip,
                    child,
                    start_time=t,
                    end_time=None,
                    length=None,
                )

                alive.append(
                    (
                        tip,
                        child,
                        state,
                    )
                )

        #
        # Death
        #
        elif event_type == "death":

            G.nodes[tip]["event"] = "death"

        #
        # Transition
        #
        else:

            _, dest = event_type

            G.nodes[tip]["event"] = "transition"

            child = next_node
            next_node += 1

            G.add_node(
                child,
                time=None,
                event=None,
                state=dest,
            )

            G.add_edge(
                tip,
                child,
                start_time=t,
                end_time=None,
                length=None,
            )

            alive.append(
                (
                    tip,
                    child,
                    dest,
                )
            )

    #
    # Truncate surviving lineages at T
    #
    for parent, tip, state in alive:

        close_branch(
            G,
            parent,
            tip,
            T,
        )

        if (np.random.random() < sampling_rates[state]):
            G.nodes[tip]["event"] = "sample"
        else:
            G.nodes[tip]["event"] = "unsampled"

    return G, root



def filter_extant_lineages(G, root):
    """
    Prune tree bottom-up to keep only extant lineages.
    Birth nodes with only one surviving child are contracted.
    """

    H = G.copy()

    # process nodes in postorder (bottom-up)
    for v in list(nx.dfs_postorder_nodes(H, source=root)):
        if v == root:
            continue

        event = H.nodes[v]["event"]
        parent = next(iter(H.predecessors(v)), None)

        # ---------- rule 1: death events ----------
        if event == "death" or event == "unsampled":
            if parent is not None:
                H.remove_edge(parent, v)
            H.remove_node(v)
            continue

        # ---------- rule 2 & 3: birth events ----------
        if event == "birth" or event == "diff" or event == "transition":
            children = list(H.successors(v))

            # no surviving children → delete birth
            if len(children) == 0:
                if parent is not None:
                    H.remove_edge(parent, v)
                H.remove_node(v)

            # exactly one surviving child → contract
            # elif len(children) == 1:
            #     c = children[0]

            #     if parent is not None:
            #         # original edges
            #         e1 = H.edges[parent, v]
            #         e2 = H.edges[v, c]

            #         # create contracted edge
            #         H.add_edge(
            #             parent,
            #             c,
            #             start_time=e1["start_time"],
            #             end_time=e2["end_time"],
            #             length=e2["end_time"] - e1["start_time"]
            #         )

            #         H.remove_edge(parent, v)

            #     H.remove_edge(v, c)
            #     H.remove_node(v)

            # two children → keep as is
            elif len(children) == 2:
                pass

            elif len(children) > 2:
                raise RuntimeError("Birth node with >2 children")

    return H


def contract_edges(G, root):

    H = nx.DiGraph()

    keep = set()

    for v in G.nodes:

        children = list(G.successors(v))

        if v == root:
            keep.add(v)

        elif len(children) != 1:
            keep.add(v)

    # copy kept nodes
    for v in keep:
        H.add_node(v, **G.nodes[v])

    transitions_list = []

    def walk_path(parent_keep, child):

        current = child

        transitions = []

        start_time = G.edges[parent_keep, child]["start_time"]

        while current not in keep:

            if G.nodes[current]["event"] in {"diff", "transition"}:

                p = next(G.predecessors(current))
                c = next(G.successors(current))

                transitions.append(
                    (
                        G.nodes[current]["time"],
                        G.nodes[p]["state"],
                        G.nodes[c]["state"]
                    )
                )

            children = list(G.successors(current))

            if len(children) == 0:
                break

            current = children[0]

        return current, transitions, start_time

    for u in keep:

        for child in G.successors(u):

            target, transitions, start_time = \
                walk_path(u, child)

            end_time = G.nodes[target]["time"]

            H.add_edge(
                u,
                target,
                start_time=start_time,
                end_time=end_time,
                length=end_time - start_time,
                transitions=transitions
            )

            for time, s1, s2 in transitions:

                transitions_list.append(
                    (
                        time,
                        u,
                        target,
                        s1,
                        s2
                    )
                )

    transitions_list.sort(key=lambda x: x[0])

    return H, transitions_list


# def filter_extant_lineages(G, root):
#     """
#     Keep only sampled extant lineages.

#     Rules
#     -----
#     - Remove death and unsampled leaves.
#     - Remove birth nodes with no surviving descendants.
#     - Contract birth nodes with exactly one surviving child.
#     - Preserve and merge transition histories stored on edges.
#     """

#     H = G.copy()

#     for v in list(nx.dfs_postorder_nodes(H, source=root)):

#         if v not in H:
#             continue

#         if v == root:
#             continue

#         parent = next(iter(H.predecessors(v)), None)
#         event = H.nodes[v]["event"]

#         # --------------------------------------------------
#         # Remove extinct / unsampled leaves
#         # --------------------------------------------------

#         if event in {"death", "unsampled"}:

#             if parent is not None and H.has_edge(parent, v):
#                 H.remove_edge(parent, v)

#             if v in H:
#                 H.remove_node(v)

#             continue

#         # --------------------------------------------------
#         # Birth node processing
#         # --------------------------------------------------

#         if event == "birth":

#             children = list(H.successors(v))

#             # ----------------------------------------------
#             # No surviving descendants
#             # ----------------------------------------------

#             if len(children) == 0:

#                 if parent is not None and H.has_edge(parent, v):
#                     H.remove_edge(parent, v)

#                 if v in H:
#                     H.remove_node(v)

#             # ----------------------------------------------
#             # Exactly one surviving child
#             # Contract:
#             #
#             # parent -> v -> child
#             #
#             # becomes
#             #
#             # parent -> child
#             # ----------------------------------------------

#             elif len(children) == 1:

#                 c = children[0]

#                 if parent is not None:

#                     e1 = H.edges[parent, v]
#                     e2 = H.edges[v, c]

#                     merged_transitions = (
#                         list(e1.get("transitions", []))
#                         + list(e2.get("transitions", []))
#                     )

#                     merged_transitions.sort(
#                         key=lambda x: x["time"]
#                     )

#                     H.add_edge(
#                         parent,
#                         c,
#                         start_time=e1["start_time"],
#                         end_time=e2["end_time"],
#                         length=(
#                             e2["end_time"]
#                             - e1["start_time"]
#                         ),
#                         start_state=e1["start_state"],
#                         end_state=e2["end_state"],
#                         transitions=merged_transitions
#                     )

#                     H.remove_edge(parent, v)

#                 H.remove_edge(v, c)
#                 H.remove_node(v)

#             # ----------------------------------------------
#             # Proper binary birth
#             # ----------------------------------------------

#             elif len(children) == 2:

#                 pass

#             else:

#                 raise RuntimeError(
#                     f"Birth node {v} has "
#                     f"{len(children)} children"
#                 )

#     # ------------------------------------------------------
#     # Special handling of root
#     # ------------------------------------------------------

#     # while root in H:

#     #     children = list(H.successors(root))

#     #     if len(children) != 1:
#     #         break

#     #     c = children[0]

#     #     e = H.edges[root, c]

#     #     # absorb root into child

#     #     H.remove_edge(root, c)
#     #     H.remove_node(root)

#     #     root = c

#     return H, root


# def contract_edges(G, root):
#     H = G.copy()
#     transitions_list = []
#     parent_dict = {}
#     children_dict = defaultdict(list)

#     for v in list(nx.dfs_postorder_nodes(H, source=root)):
#         if v == root:
#             continue

#         parent = next(iter(H.predecessors(v)), None)
#         children = list(H.successors(v))
#         event = H.nodes[v]["event"]

#         if len(children) == 1:
#             c = children[0]

#             if parent is not None:
#                 # original edges
#                 e1 = H.edges[parent, v]
#                 e2 = H.edges[v, c]

#                 # create contracted edge
#                 H.add_edge(
#                     parent,
#                     c,
#                     start_time=e1["start_time"],
#                     end_time=e2["end_time"],
#                     length=e2["end_time"] - e1["start_time"]
#                 )

#                 H.remove_edge(parent, v)

#                 if event == "diff" or event == "transition":
#                     transitions_list.append((e1["end_time"], parent, c, H.nodes[parent]["state"], H.nodes[c]["state"]))

#                     parent_dict[(parent, v)] = (parent, c)
#                     parent_dict[(v, c)] = (parent, c)
#                     children_dict[(parent, c)].append((parent, v))
#                     children_dict[(parent, c)].append((v, c))

#                     for child in children_dict[(parent, v)]:
#                         parent_dict[child] = (parent, c)

#                     for child in children_dict[(v, c)]:
#                         parent_dict[child] = (parent, c)
                    
#                     children_dict[(parent, c)].extend(children_dict[(parent, v)])
#                     children_dict[(parent, c)].extend(children_dict[(v, c)])
    
#             H.remove_edge(v, c)
#             H.remove_node(v)

#     updated_transitions_list = []
#     for time, p, c, s1, s2 in transitions_list:
#         if (p, v) in parent_dict:
#             new_p, new_c = parent_dict[(p, v)]
#             updated_transitions_list.append((time, new_p, new_c, s1, s2))
#         else:
#             updated_transitions_list.append((time, p, c, s1, s2))
                

#     return H, updated_transitions_list


def save_tree(G, root, filename):
    """
    Save a birth–death tree with all node/edge attributes to a JSON file.
    """

    data = {
        "root": root,
        "graph": json_graph.node_link_data(G)
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_tree(filename):
    """
    Load a birth–death tree saved with save_tree().
    """

    with open(filename, "r") as f:
        data = json.load(f)

    G = json_graph.node_link_graph(data["graph"], directed=True)
    root = data["root"]

    return G, root



def export_tree(T, node_file="nodes.csv", edge_file="edges.csv"):
    # --- write nodes ---
    with open(node_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "T", "is_leaf", "state_id"])

        for n, data in T.nodes(data=True):
            state = data.get("state")
            is_leaf = data.get("event") == "sample"

            # hard mapping
            if not is_leaf or state is None or state == "?":
                sid = -1
            else:
                sid = ord(state) - ord('A')

            # print(n, data)
            writer.writerow([
                n,
                float(data["T"]),
                int(is_leaf),
                sid
            ])

    # --- write edges ---
    with open(edge_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "branch_length"])

        for u, v, data in T.edges(data=True):
            writer.writerow([
                u,
                v,
                float(data.get("length", 0.0))
            ])



def main2():
    from src.plotters import plot_simulated_lineage_tree

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, help='seed [0]', default = 0)
    parser.add_argument('-o', type=str, help='output prefix', default='sample')

    args = parser.parse_args(None if sys.argv[1:] else ['-h'])

    save_dir = args.o
    seed = args.s

    diff_map_edges = [
        ("A", "B"),
        ("A", "C"),
    ]

    diff_times = [0.5, None, None]

    diff_T = nx.DiGraph()

    for i, t in enumerate(diff_times):
        diff_T.add_node(string.ascii_uppercase[i], diff_time = t)

    for s, e in diff_map_edges:
        diff_T.add_edge(s, e)

    lam = {
        "A": 5.0,
        "B": 3.0,
        "C": 1.0,
    }

    mu = {
        "A": 0.3,
        "B": 0.2,
        "C": 0.2,
    }

    types = list(diff_T.nodes)

    Qs = pd.DataFrame(0.0, index=types, columns=types)

    # A -> B, C
    Qs.loc["A", "B"] = 0.3
    Qs.loc["A", "C"] = 0.7

    G, root = simulate_multitype_bd_with_diff(
        birth_rates=lam,
        death_rates=mu,
        differentiation_times=diff_T,
        transition_probabilities=Qs,
        time_length=1.0,
        root_state="A",
        seed=seed
    )

    os.makedirs(save_dir, exist_ok=True)

    print(G.nodes(data=True))
   
    save_tree(G, root, f"{save_dir}/T.json")
    G_extant = filter_extant_lineages(G, root)
    save_tree(G_extant, root, f"{save_dir}/T_reconstructed.json")


    try:
        plot_simulated_lineage_tree(G, root)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root)
        plt.savefig(f"{save_dir}/T_reconstructed.png")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")



def main3():
    from src.plotters import plot_simulated_lineage_tree

    G, root = simulate_multitype_bd_ctmc(
        birth_rates={
            "A": 7.0,
            "B": 5.5,
            "C": 3.5,
        },
        death_rates={
            "A": 0.2,
            "B": 0.1,
            "C": 0.1,
        },
        sampling_rates={
            "A": 0.5,
            "B": 0.5,
            "C": 0.5,
        },
        Q=pd.DataFrame(
            [
                [0.0, 1.0, 1.5],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0],
            ],
            index=["A","B","C"],
            columns=["A","B","C"]
        ),
        T=1.0,
        root_state="A",
        seed=0
    )

    save_dir = "/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/0"

    os.makedirs(save_dir, exist_ok=True)

    print(G.nodes(data=True))
   
    save_tree(G, root, f"{save_dir}/T.json")
    G_extant = filter_extant_lineages(G, root)
    save_tree(G_extant, root, f"{save_dir}/T_extant.json")

    export_tree(G_extant, f"{save_dir}/nodes.csv", f"{save_dir}/edges.csv")


    try:
        plot_simulated_lineage_tree(G, root)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def main4():
    from src.plotters import plot_simulated_lineage_tree

    G, root = simulate_multitype_bdt_ctmc_window(
        birth_rates={
            "A": 7.0,
            "B": 5.5,
            "C": 3.5,
        },
        death_rates={
            "A": 0.2,
            "B": 0.1,
            "C": 0.1,
        },
        sampling_rates={
            "A": 0.5,
            "B": 0.5,
            "C": 0.5,
        },
        Q=pd.DataFrame(
            [
                [0.0, 4.0, 6.0],
                [0.0, 0.0, 3.2],
                [0.0, 0.0, 0.0],
            ],
            index=["A","B","C"],
            columns=["A","B","C"]
        ),
        transition_windows = {
            ("A", "B"): [(0.55, 0.7)],
            ("A", "C"): [(0.55, 0.7)],
            ("B", "C"): [(0.7, 0.9)],
        },
        T=1.0,
        root_state="A",
        seed=0
    )

    save_dir = "/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc_window/0"

    os.makedirs(save_dir, exist_ok=True)

    print(G.nodes(data=True))
   
    save_tree(G, root, f"{save_dir}/T.json")
    G_extant = filter_extant_lineages(G, root)
    save_tree(G_extant, root, f"{save_dir}/T_extant.json")

    export_tree(G_extant, f"{save_dir}/nodes.csv", f"{save_dir}/edges.csv")


    try:
        plot_simulated_lineage_tree(G, root)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def main5():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    num_setting = 0

    with open(f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/settings.csv", "w") as f:

        print("setting,birth(A),birth(B),birth(C),transition(A>B),transition(B>C)", file=f, flush=True)

        for ba in [4, 5, 6, 7, 8]:
            for bb in [4, 5, 6, 7, 8]:
                for bc in [4, 5, 6, 7, 8]:
                    for abt in [1, 2, 4]:
                        for bct in [1, 2, 4]:
                            print(num_setting, ba, bb, bc, abt, bct, file=f, flush=True, sep=',')
                            # for seed in range(32):
                                
                            #     print(num_setting, ba, bb, bc, abt, bct, seed, flush=True)

                            #     G, root = simulate_multitype_bdt_ctmc_window(
                            #         birth_rates={
                            #             "A": ba,
                            #             "B": bb,
                            #             "C": bc,
                            #         },
                            #         death_rates={
                            #             "A": 0,
                            #             "B": 0,
                            #             "C": 0,
                            #         },
                            #         sampling_rates={
                            #             "A": 0.5,
                            #             "B": 0.5,
                            #             "C": 0.5,
                            #         },
                            #         Q=pd.DataFrame(
                            #             [
                            #                 [0.0, abt, 0.0],
                            #                 [0.0, 0.0, bct],
                            #                 [0.0, 0.0, 0.0],
                            #             ],
                            #             index=["A","B","C"],
                            #             columns=["A","B","C"]
                            #         ),
                            #         transition_windows = {
                            #             ("A", "B"): [(0, 1)],
                            #             ("B", "C"): [(0, 1)],
                            #         },
                            #         T=1.0,
                            #         root_state="A",
                            #         seed=seed
                            #     )

                            #     save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{num_setting}"

                            #     os.makedirs(save_dir, exist_ok=True)

                            #     # print(G.nodes(data=True))
                            
                            #     G = add_node_times(G)
                            
                            #     # save_tree(G, root, f"{save_dir}/T_{seed}.json")
                            #     G_extant = filter_extant_lineages(G, root)

                            #     # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

                            #     export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")

                            #     try:
                            #         plot_simulated_lineage_tree(G, root)
                            #         plt.savefig(f"{save_dir}/T_{seed}.png")
                            #         plt.close()

                            #         plot_simulated_lineage_tree(G_extant, root)
                            #         plt.savefig(f"{save_dir}/T_extant_{seed}.png")
                            #         plt.close()
                            #     except Exception as e:
                            #         print(f"Plotting failed: {e}")

                            num_setting += 1


def main6():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args(None if sys.argv[1:] else ['-h'])

    setting = args.setting
    seed = args.seed

    df = pd.read_csv("/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/settings.csv", index_col=0)

    row = df.loc[setting]

    ba = row.get("birth(A)")
    bb = row.get("birth(B)")
    bc = row.get("birth(C)")
    abt = row.get("transition(A>B)")
    bct = row.get("transition(B>C)")

    print(setting, ba, bb, bc, abt, bct, seed, flush=True)

    G, root = simulate_multitype_bdt_ctmc_window(
        birth_rates={
            "A": ba,
            "B": bb,
            "C": bc,
        },
        death_rates={
            "A": 0,
            "B": 0,
            "C": 0,
        },
        sampling_rates={
            "A": 0.5,
            "B": 0.5,
            "C": 0.5,
        },
        Q=pd.DataFrame(
            [
                [0.0, abt, 0.0],
                [0.0, 0.0, bct],
                [0.0, 0.0, 0.0],
            ],
            index=["A","B","C"],
            columns=["A","B","C"]
        ),
        transition_windows = {
            ("A", "B"): [(0, 1)],
            ("B", "C"): [(0, 1)],
        },
        T=1.0,
        root_state="A",
        seed=seed
    )

    save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

    os.makedirs(save_dir, exist_ok=True)

    # print(G.nodes(data=True))

    G = add_node_times(G)

    # save_tree(G, root, f"{save_dir}/T_{seed}.json")
    G_extant = filter_extant_lineages(G, root)

    # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

    export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")

    try:
        plot_simulated_lineage_tree(G, root)
        plt.savefig(f"{save_dir}/T_{seed}.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root)
        plt.savefig(f"{save_dir}/T_extant_{seed}.png")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def main7():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 6.5
    bb = 9
    bc = 10
    abt = 1.0
    bct = 1.5
    sampling_rate = 0.5

    for seed in range(25):
        # seed = 0

        setting = f"test_{seed}"

        print(setting)

        
        G, root = simulate_multitype_bd_ctmc(
            birth_rates={
                "A": ba,
                "B": bb,
                "C": bc,
            },
            death_rates={
                "A": 0,
                "B": 0,
                "C": 0,
            },
            sampling_rates={
                "A": sampling_rate,
                "B": sampling_rate,
                "C": sampling_rate,
            },
            Q=pd.DataFrame(
                [
                    [0.0, abt, 0.0],
                    [0.0, 0.0, bct],
                    [0.0, 0.0, 0.0],
                ],
                index=["A","B","C"],
                columns=["A","B","C"]
            ),
            T=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)

        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)

        G_extant, transitions = contract_edges(G_extant, root)

        print(f"Number of transitions: {len(transitions)}")

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_extant, f"{save_dir}/nodes_0.csv", f"{save_dir}/edges_0.csv")

        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

        with open(f'{save_dir}/transition_times_with_edges.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for t, e1, e2, s1, s2 in sorted(transitions):
                print(f"{t},{e1},{e2},{s1},{s2}", file=f, flush=True)



def main8():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 6.5
    sampling_rate = 0.5

    for seed in range(25):
        # seed = 0

        setting = f"test_single_{seed}"

        print(setting)

        
        G, root = simulate_multitype_bd_ctmc(
            birth_rates={
                "A": ba,
            },
            death_rates={
                "A": 0,
            },
            sampling_rates={
                "A": sampling_rate,
            },
            Q=pd.DataFrame([[0.0],],
                index=["A"],
                columns=["A"]
            ),
            T=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)

        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)

        G_contract, transitions = contract_edges(G_extant, root)

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_contract, f"{save_dir}/nodes_0.csv", f"{save_dir}/edges_0.csv")

        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

def main():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 6
    bb = 7
    abt = 0.5
    sampling_rate = 0.5

    for seed in range(25):
        # seed = 0

        setting = f"test_two_{seed}"

        print(setting)

        
        G, root = simulate_multitype_bd_ctmc(
            birth_rates={
                "A": ba,
                "B": bb,
            },
            death_rates={
                "A": 0,
                "B": 0,
            },
            sampling_rates={
                "A": sampling_rate,
                "B": sampling_rate,
            },
            Q=pd.DataFrame([[0.0, abt], [0.0, 0.0]],
                index=["A", "B"],
                columns=["A", "B"]
            ),
            T=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)
        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)

        G_contract, transitions = contract_edges(G_extant, root)

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_contract, f"{save_dir}/nodes_0.csv", f"{save_dir}/edges_0.csv")

        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

        
        with open(f'{save_dir}/transition_times_with_edges.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, e1, e2, s1, s2 in sorted(transitions):
                print(a, e1, e2, s1, s2, sep=',', file=f, flush=True)


def main9():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 6.5
    bb = 9
    bc = 10
    abt = 1.0
    bct = 1.5
    sampling_rate = 0.5

    for seed in range(1):
        # seed = 0

        setting = f"test2_{seed}"

        print(setting)

        
        G, root = simulate_multitype_bd_ctmc(
            birth_rates={
                "A": ba,
                "B": bb,
                "C": bc,
            },
            death_rates={
                "A": 0,
                "B": 0,
                "C": 0,
            },
            sampling_rates={
                "A": sampling_rate,
                "B": sampling_rate,
                "C": sampling_rate,
            },
            Q=pd.DataFrame(
                [
                    [0.0, abt, 0.0],
                    [0.0, 0.0, bct],
                    [0.0, 0.0, 0.0],
                ],
                index=["A","B","C"],
                columns=["A","B","C"]
            ),
            T=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)

        print(len(G.nodes))

        export_tree(G, f"{save_dir}/nodes_0.csv", f"{save_dir}/edges_0.csv")

        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_extant, f"{save_dir}/nodes_0.csv", f"{save_dir}/edges_0.csv")

        transitions = []
        # for u, v, data in G_extant.edges(data=True):
        #     for trans in data.get("transitions", []):
        #         transitions.append((
        #             trans["time"],
        #             u,
        #             v,
        #             trans["from"],
        #             trans["to"]
        #         ))

        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

        with open(f'{save_dir}/transition_times_with_edges.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for t, e1, e2, s1, s2 in sorted(transitions):
                print(f"{t},{e1},{e2},{s1},{s2}", file=f, flush=True)




def main10():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 7
    bb = 9
    abp = 0.5
    transition_time = 0.6
    sampling_rate = 0.5

    diff_T = nx.DiGraph()
    diff_T.add_node("A", transition_times = [transition_time])
    diff_T.add_node("B", transition_times = [])
    diff_T.add_edge("A", "B")

    diff_probs = pd.DataFrame(
        [
            [1 - abp, abp],
            [0.0, 0.0],
        ],
        index=["A", "B"],
        columns=["A", "B"]
    )


    for seed in range(100):

        setting = f"test4_multi"

        print(setting)

        
        G, root = simulate_multitype_bd_with_diff(
            birth_rates={
                "A": ba,
                "B": bb,
            },
            death_rates={
                "A": 0,
                "B": 0,
            },
            sampling_rates={
                "A": sampling_rate,
                "B": sampling_rate,
            },
            transition_probabilities=diff_probs,
            differentiation_times=diff_T,
            time_length=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)

        print(len(G.nodes))

        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)
        G_extant, transitions = contract_edges(G_extant, root)

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")


        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_{seed}.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant_{seed}.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times_{seed}.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

        with open(f'{save_dir}/transition_times_with_edges_{seed}.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for t, e1, e2, s1, s2 in sorted(transitions):
                print(f"{1-t},{e1},{e2},{ord(s1) - ord('A')},{ord(s2) - ord('A')}", file=f, flush=True)





def main10():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 7
    bb = 7
    abp = 0.8
    transition_time = 0.4
    sampling_rate = 0.5

    diff_T = nx.DiGraph()
    diff_T.add_node("A", transition_times = [transition_time])
    diff_T.add_node("B", transition_times = [])
    diff_T.add_edge("A", "B")

    diff_probs = pd.DataFrame(
        [
            [1 - abp, abp],
            [0.0, 0.0],
        ],
        index=["A", "B"],
        columns=["A", "B"]
    )


    for seed in range(100):

        setting = f"test5_multi"

        print(setting)

        
        G, root = simulate_multitype_bd_with_diff(
            birth_rates={
                "A": ba,
                "B": bb,
            },
            death_rates={
                "A": 0,
                "B": 0,
            },
            sampling_rates={
                "A": sampling_rate,
                "B": sampling_rate,
            },
            transition_probabilities=diff_probs,
            differentiation_times=diff_T,
            time_length=1.0,
            root_state="A",
            seed=seed
        )

        save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"

        os.makedirs(save_dir, exist_ok=True)

        # print(G.nodes(data=True))

        G = add_node_times(G)

        print(len(G.nodes))

        # save_tree(G, root, f"{save_dir}/T_{seed}.json")
        G_extant = filter_extant_lineages(G, root)
        G_extant, transitions = contract_edges(G_extant, root)

        # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

        export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")


        # try:
        plot_simulated_lineage_tree(G, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_{seed}.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root, show_labels=False)
        plt.savefig(f"{save_dir}/T_extant_{seed}.png")
        plt.close()
        # except Exception as e:
        #     print(f"Plotting failed: {e}")

        with open(f'{save_dir}/transition_times_{seed}.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for a, _, _, _, _ in sorted(transitions):
                print(a, file=f, flush=True)

        with open(f'{save_dir}/transition_times_with_edges_{seed}.csv', 'w') as f:
            print(len(transitions), file=f, flush=True)
            for t, e1, e2, s1, s2 in sorted(transitions):
                print(f"{1-t},{e1},{e2},{ord(s1) - ord('A')},{ord(s2) - ord('A')}", file=f, flush=True)





def main11():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    ba = 9
    bb = 8
    bc = 7
    abp = 0.35
    abc = 0.45
    transition_time = 0.4
    sampling_rates = [0.5, 0.4, 0.6]

    diff_T = nx.DiGraph()
    diff_T.add_node("A", transition_times = [transition_time])
    diff_T.add_node("B", transition_times = [])
    diff_T.add_node("C", transition_times = [])
    diff_T.add_edge("A", "B")
    diff_T.add_edge("A", "C")

    diff_probs = pd.DataFrame(
        [
            [1 - abp - abc, abp, abc],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"]
    )

    for multi in range(5):

        for seed in range(25):

            setting = f"test{multi+6}_multi"

            print(setting)

            
            G, root = simulate_multitype_bd_with_diff(
                birth_rates={
                    "A": ba,
                    "B": bb,
                    "C": bc,
                },
                death_rates={
                    "A": 0.2,
                    "B": 0.7,
                    "C": 0.5,
                },
                sampling_rates={
                    "A": sampling_rates[0],
                    "B": sampling_rates[1],
                    "C": sampling_rates[2],
                },
                transition_probabilities=diff_probs,
                differentiation_times=diff_T,
                time_length=1.0,
                root_state="A",
                seed=multi * 25 + seed
            )

            save_dir = f"/home/samin/workspace/research/phylodynamic-CARTA/data/simulation/bdt_ctmc/{setting}"

            os.makedirs(save_dir, exist_ok=True)

            # print(G.nodes(data=True))

            G = add_node_times(G)

            print(len(G.nodes))

            # save_tree(G, root, f"{save_dir}/T_{seed}.json")
            G_extant = filter_extant_lineages(G, root)
            G_extant, transitions = contract_edges(G_extant, root)

            # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

            export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")


            # try:
            plot_simulated_lineage_tree(G, root, show_labels=False)
            plt.savefig(f"{save_dir}/T_{seed}.png")
            plt.close()

            plot_simulated_lineage_tree(G_extant, root, show_labels=False)
            plt.savefig(f"{save_dir}/T_extant_{seed}.png")
            plt.close()
            # except Exception as e:
            #     print(f"Plotting failed: {e}")

            with open(f'{save_dir}/transition_times_{seed}.csv', 'w') as f:
                print(len(transitions), file=f, flush=True)
                for a, _, _, _, _ in sorted(transitions):
                    print(a, file=f, flush=True)

            with open(f'{save_dir}/transition_times_with_edges_{seed}.csv', 'w') as f:
                print(len(transitions), file=f, flush=True)
                for t, e1, e2, s1, s2 in sorted(transitions):
                    print(f"{1-t},{e1},{e2},{ord(s1) - ord('A')},{ord(s2) - ord('A')}", file=f, flush=True)




def main12():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    abp = 0.5
    transition_times = [0.5]
    birth_rates = [8, 7]
    death_rates = [0.2, 0.5]
    sampling_rates = [0.5, 0.6]

    diff_T = nx.DiGraph()
    diff_T.add_node("A", transition_times = transition_times)
    diff_T.add_node("B", transition_times = [])
    diff_T.add_edge("A", "B")

    diff_probs = pd.DataFrame(
        [
            [1 - abp, abp],
            [0.0, 1.0],
        ],
        index=["A", "B"],
        columns=["A", "B"]
    )

    num_trees_per_setting = 1000

    for multi in range(5):

        for seed in range(num_trees_per_setting):

            setting = f"test{multi+6}_multi"

            inc = 0

            while True:
                print(setting, seed, inc)


                try:

                    G, root = simulate_multitype_bd_with_diff(
                        birth_rates={
                            "A": birth_rates[0],
                            "B": birth_rates[1],
                        },
                        death_rates={
                            "A": death_rates[0],
                            "B": death_rates[1],
                        },
                        sampling_rates={
                            "A": sampling_rates[0],
                            "B": sampling_rates[1],
                        },
                        transition_probabilities=diff_probs,
                        differentiation_times=diff_T,
                        time_length=1.0,
                        root_state="A",
                        seed=(multi * num_trees_per_setting + seed) * 10 + inc
                    )
        
                    save_dir = f"/Users/samin/workspace/phylodynamic-carta/data/simulation/coord_diff/{setting}"
        
                    os.makedirs(save_dir, exist_ok=True)
        
                    # print(G.nodes(data=True))
        
                    G = add_node_times(G)
        
                    print(len(G.nodes))
        
                    # save_tree(G, root, f"{save_dir}/T_{seed}.json")
                    G_extant = filter_extant_lineages(G, root)
                    G_extant, transitions = contract_edges(G_extant, root)
        
                    # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")
        
                    export_tree(G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv")
        
        
                    # try:
                    plot_simulated_lineage_tree(G, root, show_labels=False)
                    plt.savefig(f"{save_dir}/T_{seed}.png")
                    plt.close()
        
                    plot_simulated_lineage_tree(G_extant, root, show_labels=False)
                    plt.savefig(f"{save_dir}/T_extant_{seed}.png")
                    plt.close()
                    # except Exception as e:
                    #     print(f"Plotting failed: {e}")
        
                    with open(f'{save_dir}/transition_times_{seed}.csv', 'w') as f:
                        print(len(transitions), file=f, flush=True)
                        for a, _, _, _, _ in sorted(transitions):
                            print(a, file=f, flush=True)
        
                    with open(f'{save_dir}/transition_times_with_edges_{seed}.csv', 'w') as f:
                        print(len(transitions), file=f, flush=True)
                        for t, e1, e2, s1, s2 in sorted(transitions):
                            print(f"{1-t},{e1},{e2},{ord(s1) - ord('A')},{ord(s2) - ord('A')}", file=f, flush=True)
        
                    break

                except Exception as e:

                    inc += 1

        



            



if __name__ == "__main__":
    main12()
