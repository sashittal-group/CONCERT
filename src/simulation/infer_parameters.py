import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import os
import json
import argparse
from src.plotters import plot_differentiation_tree, plot_simulated_lineage_tree
from src.simulator import (
    simulate_multitype_bd_with_diff,
    save_tree,
    filter_extant_lineages,
    load_tree,
)
from src.likelihood import add_node_times, assign_states_to_internal_nodes

from math import exp, log
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from decimal import Decimal, getcontext


MAX_BIRTH = 50.0


def calculate_log_likelihood_2(
    T: nx.DiGraph,
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities: pd.DataFrame,
    diff_T: nx.DiGraph,
):

    states = list(diff_T.nodes())
    U = dict()

    for state in states:
        U[(state, 0)] = 1 - sampling_rate

    for diff_node in nx.dfs_postorder_nodes(diff_T):
        diff_time = diff_T.nodes[diff_node]["diff_time"]
        if diff_time is None:
            diff_time = 0

        if diff_time > 0:
            U_t = 0
            children = list(diff_T.successors(diff_node))
            for child in children:
                child_node = diff_T.nodes[child]
                child_diff_time = child_node["diff_time"]
                if child_diff_time is None:
                    child_diff_time = 0

                U_child = U[(child, child_diff_time)]

                C = (U_child - 1) / (
                    U_child - (death_rates[child] / birth_rates[child])
                )

                U_child_s = 1 - (birth_rates[child] - death_rates[child]) / (
                    birth_rates[child]
                    * (
                        1
                        - (1 / C)
                        * exp(
                            -(birth_rates[child] - death_rates[child])
                            * (diff_time - child_diff_time)
                        )
                    )
                )

                tp = transition_probabilities.loc[diff_node, child]

                U_t += U_child_s * tp

            U[(diff_node, diff_time)] = U_t

    ll = 0.0
    for n in T.nodes():
        ll += log(birth_rates[T.nodes[n]["state"]])

    for u, v in T.edges():
        u_t = T.nodes[u]["T"]
        v_t = T.nodes[v]["T"]
        assert u_t > v_t

        u_state = T.nodes[u]["state"]
        v_state = T.nodes[v]["state"]

        state_path = nx.shortest_path(diff_T, source=u_state, target=v_state)

        state = state_path[0]
        state_diff_time = diff_T.nodes[state]["diff_time"]
        if state_diff_time is None:
            state_diff_time = 0

        U_t = U[(state, state_diff_time)]
        C = (U_t - 1) / (
            U_t - (death_rates[state] / birth_rates[state])
        )  # TODO(Samin): Is this correct? Seems so

        e_t = max(state_diff_time, v_t)
        flow = (
            exp(-(birth_rates[state] - death_rates[state]) * (u_t - e_t))
            * (
                (
                    C
                    - exp(
                        -(birth_rates[state] - death_rates[state])
                        * (e_t - state_diff_time)
                    )
                )
                / (
                    C
                    - exp(
                        -(birth_rates[state] - death_rates[state])
                        * (u_t - state_diff_time)
                    )
                )
            )
            ** 2
        )

        assert not np.isnan(flow)

        ll += log(flow)

        for i in range(1, len(state_path)):
            state = state_path[i]
            prev_state = state_path[i - 1]

            state_diff_time = diff_T.nodes[state]["diff_time"]
            if state_diff_time is None:
                state_diff_time = 0

            prev_state_diff_time = diff_T.nodes[prev_state]["diff_time"]
            if prev_state_diff_time is None:
                prev_state_diff_time = 0

            U_t = U[(state, state_diff_time)]
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]))

            e_t = max(state_diff_time, v_t)
            flow = (
                exp(
                    -(birth_rates[state] - death_rates[state])
                    * (prev_state_diff_time - e_t)
                )
                * (
                    (
                        C
                        - exp(
                            -(birth_rates[state] - death_rates[state])
                            * (e_t - state_diff_time)
                        )
                    )
                    / (
                        C
                        - exp(
                            -(birth_rates[state] - death_rates[state])
                            * (prev_state_diff_time - state_diff_time)
                        )
                    )
                )
                ** 2
            )

            assert not np.isnan(flow)

            ll += log(flow)

        ## transitions
        for i in range(1, len(state_path)):
            state = state_path[i]
            prev_state = state_path[i - 1]

            ll += log(transition_probabilities.loc[prev_state, state])

    return ll


def calculate_log_likelihood_graph_diff(
    T: nx.DiGraph,
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities: pd.DataFrame,
    diff_T: nx.DiGraph,
):

    states = list(diff_T.nodes())
    U = dict()

    for state in states:
        U[(state, 0)] = 1 - sampling_rate

    for diff_node in nx.dfs_postorder_nodes(diff_T):
        diff_time = diff_T.nodes[diff_node]["diff_time"]
        if diff_time is None:
            diff_time = 0

        if diff_time > 0:
            U_t = 0
            children = list(diff_T.successors(diff_node))
            for child in children:
                child_node = diff_T.nodes[child]
                child_diff_time = child_node["diff_time"]
                if child_diff_time is None:
                    child_diff_time = 0

                U_child = U[(child, child_diff_time)]

                C = (U_child - 1) / (
                    U_child - (death_rates[child] / birth_rates[child])
                )

                U_child_s = 1 - (birth_rates[child] - death_rates[child]) / (
                    birth_rates[child]
                    * (
                        1
                        - (1 / C)
                        * exp(
                            -(birth_rates[child] - death_rates[child])
                            * (diff_time - child_diff_time)
                        )
                    )
                )

                tp = transition_probabilities.loc[diff_node, child]

                U_t += U_child_s * tp

            U[(diff_node, diff_time)] = U_t

    ctx = getcontext()
    ctx.prec = 50
    ctx.Emax = 99999999
    ctx.Emin = -99999999

    LL = dict()

    def calculate_log_likelihood_along_edge(u, v, t, state):
        key = (u, v, t, state)
        if key in LL:
            return LL[key]

        diff_time = diff_T.nodes.get(state, {}).get("diff_time")
        if diff_time is None:
            diff_time = 0

        v_node = T.nodes[v]
        v_t = v_node["T"]

        if v_t < diff_time < t:  # Change of state within the edge

            U_t = U[(state, diff_time)]
            ## TODO Fix the issue that requires the + 1e-200
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]) + 1e-200)

            flow = (
                exp(-(birth_rates[state] - death_rates[state]) * (t - diff_time))
                * (
                    (C - 1)
                    / (
                        C
                        - exp(
                            -(birth_rates[state] - death_rates[state]) * (t - diff_time)
                        )
                    )
                )
                ** 2
            )

            p = Decimal(0)
            for child_state in diff_T.successors(state):
                left_seg_ll = Decimal(
                    calculate_log_likelihood_along_edge(u, v, diff_time, child_state)
                )
                # print("exp: ", left_seg_ll)
                left_seg = left_seg_ll.exp()
                p += (
                    Decimal(transition_probabilities.loc[state, child_state]) * left_seg
                )

            LL[key] = float(
                Decimal(flow).ln() + p.ln()
            )  ## Why was ln not applied to p?

            # assert not np.isnan(LL[key])

        else:  # Edge ends with boundary condition

            U_t = U[(state, diff_time)]
            ## TODO Fix the issue that requires the + 1e-200
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]) + 1e-200)
            flow = (
                exp(-(birth_rates[state] - death_rates[state]) * (t - v_t))
                * (
                    (
                        C
                        - exp(
                            -(birth_rates[state] - death_rates[state])
                            * (v_t - diff_time)
                        )
                    )
                    / (
                        C
                        - exp(
                            -(birth_rates[state] - death_rates[state]) * (t - diff_time)
                        )
                    )
                )
                ** 2
            )

            children = list(T.successors(v))

            if len(children) == 0:  # Boundary condition: current time
                v_state = v_node["state"]
                if state == v_state or v_state == "?":
                    p = sampling_rate
                    LL[key] = log(flow) + log(p)
                else:
                    p = 0
                    LL[key] = -np.inf
                # assert not np.isnan(LL[key]), (flow, p, C, U_t, birth_rates[state], death_rates[state], (U_t - 1), (U_t - (death_rates[state] / birth_rates[state]) + 1e-200))

            elif len(children) == 2:  # Boundary condition: birth event
                LL1 = calculate_log_likelihood_along_edge(v, children[0], v_t, state)
                LL2 = calculate_log_likelihood_along_edge(v, children[1], v_t, state)
                LL[key] = log(flow) + LL1 + LL2 + log(birth_rates[state])

                # assert not np.isnan(LL[key])

            else:
                raise Exception("#Children must be 0 or 2")

        # assert not np.isnan(LL[key])

        return LL[key]

    for parent in nx.dfs_postorder_nodes(T):
        node = T.nodes[parent]
        node_time = node["T"]
        node_state = node.get("state")
        children = list(T.successors(parent))

        for child in children:
            if node_state is not None and node_state != "?":
                calculate_log_likelihood_along_edge(
                    parent, child, node_time, node_state
                )
            else:
                for state in states:
                    calculate_log_likelihood_along_edge(parent, child, node_time, state)

    root_node = [n for n, d in T.in_degree() if d == 0][0]
    root_child = next(T.successors(root_node))
    root_time = T.nodes[root_node]["T"]
    diff_T_root = [n for n, d in diff_T.in_degree() if d == 0][0]

    return LL[(root_node, root_child, root_time, diff_T_root)]


def calculate_log_likelihood(T, birth_rate, death_rate, sampling_rate=1):
    ll = 0.0

    ll += len(T.nodes()) * log(birth_rate)

    diff_time = 0
    U = 1 - sampling_rate
    C = (U - 1) / (U - (death_rate / birth_rate))

    for u, v in T.edges():
        u_t = T.nodes[u]["T"]
        v_t = T.nodes[v]["T"]
        assert u_t > v_t

        flow = (
            exp(-(birth_rate - death_rate) * (u_t - v_t))
            * (
                (C - exp(-(birth_rate - death_rate) * (v_t - diff_time)))
                / (C - exp(-(birth_rate - death_rate) * (u_t - diff_time)))
            )
            ** 2
        )

        assert not np.isnan(flow)

        ll += log(flow)

    return ll


def unpack_params(theta, diff_T):

    # TODO (Samin): Make this global to reduce computation
    types = sorted(list(diff_T.nodes()))
    types_unq = sorted(list(set([t.split("_")[0] for t in types])))

    transition_maps = [
        (i, succ) for i in diff_T.nodes() if (succ := list(diff_T.successors(i)))
    ]

    k = 0
    birth_rates = {}
    death_rates = {}

    for tu in types_unq:
        for t in types:
            if t.split("_")[0] == tu:
                birth_rates[t] = MAX_BIRTH * sigmoid(theta[k])
        k += 1

    for t in types:
        death_rates[t] = 1e-50

    Q = pd.DataFrame(0.0, index=types, columns=types)
    for i, transition_to in transition_maps:

        logits = []
        js = []

        for j in transition_to:
            logits.append(theta[k])
            js.append(j)
            k += 1

        logits = np.array(logits)

        # stable softmax
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        for idx, j in enumerate(js):
            Q.loc[i, j] = probs[idx]

    return birth_rates, death_rates, Q


def neg_log_likelihood(
    theta, log_likelihood_fn, cell_lineage_trees, diff_T, sampling_rate=1.0
):
    birth_rates, death_rates, Q = unpack_params(theta, diff_T)

    l = 0
    for cell_lineage_T in cell_lineage_trees:
        l += log_likelihood_fn(
            T=cell_lineage_T,
            birth_rates=birth_rates,
            death_rates=death_rates,
            sampling_rate=sampling_rate,
            transition_probabilities=Q,
            diff_T=diff_T,
        )

    return -l


def pack_initial(diff_T):
    types = sorted(list(diff_T.nodes()))
    types_unq = sorted(list(set([t.split("_")[0] for t in types])))

    transition_maps = [
        (i, succ) for i in diff_T.nodes() if (succ := list(diff_T.successors(i)))
    ]

    params = []

    for t in types_unq:
        params.append(0.0)  # sigmoid(0)=0.5 → birth ≈ halfway to MAX_BIRTH

    # transition logits
    for i, transition_to in transition_maps:
        for j in transition_to:
            params.append(0.0)

    return np.array(params)


def find_params(log_likelihood_fn, trees, diff_T):

    theta0 = pack_initial(diff_T)

    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(log_likelihood_fn, trees, diff_T, 1.0),
        method="L-BFGS-B",
    )

    birth_rates, death_rates, Q = unpack_params(result.x, diff_T)

    return birth_rates, death_rates, Q


def load_and_process_tree(path):
    T, root = load_tree(f"{path}/T_extant.json")

    cell_lineage_T = nx.DiGraph()

    for u, v, data in T.edges(data=True):
        cell_lineage_T.add_edge(u, v, weight=data["length"])

    for n, d in T.nodes(data=True):
        if d["event"] == "sample":
            cell_lineage_T.nodes[n]["state"] = d["type"]

    cell_lineage_T = add_node_times(cell_lineage_T, reverse=False)
    T = cell_lineage_T
    rev_T = T.copy()
    for n in T.nodes():
        rev_T.nodes[n]["T"] = (T.nodes[n]["T"]) / 5

    for u, v, data in rev_T.edges(data=True):
        data["weight"] /= 5

    rev_T2 = rev_T.copy()

    for n in rev_T.nodes():
        rev_T2.nodes[n]["T"] = 1 - rev_T.nodes[n]["T"]

    return rev_T2


def single_tree_loader(path):
    T = load_and_process_tree(path=path)
    return [T]


def multi_tree_loader(paths):
    Ts = []
    for path in paths:
        Ts.append(load_and_process_tree(path=path))
    return Ts


def diff_T_loader(path):
    diff_T, _ = load_tree(f"{path}/../diff_T.json")
    corrected_diff_T = diff_T.copy()
    for n in diff_T.nodes():
        if diff_T.nodes[n]["diff_time"] is None:
            corrected_diff_T.nodes[n]["diff_time"] = 0
        else:
            corrected_diff_T.nodes[n]["diff_time"] = (
                1 - diff_T.nodes[n]["diff_time"] / 5
            )

    return corrected_diff_T


def output_fn(path, birth_rates, death_rates, Q):

    dct = {
        "birth_rates": birth_rates,
    }

    os.makedirs(f"{path}/birth_given_diff_times", exist_ok=True)
    with open(f"{path}/birth_given_diff_times/inferred_parameters.json", "w") as f:
        json.dump(dct, fp=f, indent=4)

    Q.to_csv(f"{path}/birth_given_diff_times/inferred_q.csv")


def predict_params_given_diff_times(
    log_likelihood_fn, trees, corrected_diff_T, preassign_internal_nodes
):

    if preassign_internal_nodes:
        for i in range(len(trees)):
            trees[i] = assign_states_to_internal_nodes(
                trees[i], corrected_diff_T, correct=True
            )

    birth_rates, death_rates, Q = find_params(
        log_likelihood_fn=log_likelihood_fn, trees=trees, diff_T=corrected_diff_T
    )

    return birth_rates, death_rates, Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to a single tree")
    parser.add_argument(
        "--paths", type=str, nargs="+", help="List of paths to multiple trees"
    )
    parser.add_argument("--graph", action="store_true")

    args = parser.parse_args()

    print(args)

    # Determine which argument is provided
    if args.paths:

        paths = [os.path.dirname(p) for p in args.paths]
        trees = multi_tree_loader(paths=paths)
        corrected_diff_T = diff_T_loader(path=paths[0])

        outpath = (
            str(os.path.dirname(paths[0]))
            .replace("tree", f"tree_comb_{len(paths)}")
            .replace("2_stage", f"2_stage_comb_{len(paths)}")
            + "/"
            + "_".join([os.path.basename(p) for p in paths])
        )

    elif args.path:
        trees = single_tree_loader(path=args.path)
        corrected_diff_T = diff_T_loader(path=args.path)

        outpath = args.path

    else:
        parser.error("You must provide either --path or --paths")

    log_likelihood_fn = calculate_log_likelihood_2
    preassign_internal_nodes = True
    if args.graph:
        log_likelihood_fn = calculate_log_likelihood_graph_diff
        preassign_internal_nodes = False

    birth_rates, death_rates, Q = predict_params_given_diff_times(
        log_likelihood_fn=log_likelihood_fn,
        trees=trees,
        corrected_diff_T=corrected_diff_T,
        preassign_internal_nodes=preassign_internal_nodes,
    )

    output_fn(outpath, birth_rates, death_rates, Q)
