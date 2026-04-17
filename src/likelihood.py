from math import exp, log
import pandas as pd
import numpy as np
import networkx as nx
from decimal import Decimal, getcontext



def calculate_likelihood_of_tree(
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities: pd.DataFrame,
    cell_lineage_T,
    diff_T,
    root_time=1.0,
):
    
    states = list(diff_T.nodes())
    L = dict()
    U = dict()

    for state in states:
        U[(state, 0)] = 1 - sampling_rate
    
    for diff_node in nx.dfs_postorder_nodes(diff_T):
        # TODO(Samin): Make this reversal in a good place. Will forget soon!!
        diff_time = diff_T.nodes[diff_node]['diff_time']
        if diff_time is None: diff_time = 0
        # else: diff_time = 1 - diff_time

        if diff_time > 0:
            U_t = 0
            children = list(diff_T.successors(diff_node))
            for child in children:
                child_node = diff_T.nodes[child]
                child_diff_time = child_node['diff_time']
                if child_diff_time is None: child_diff_time = 0
                # else: child_diff_time = 1 - child_diff_time

                U_child = U[(child, child_diff_time)]

                C = (U_child - 1) / (U_child - (death_rates[child] / birth_rates[child]) + 1e-50)
                
                U_child_s = 1 - (birth_rates[child] - death_rates[child]) / \
                    (birth_rates[child] * (1 - (1 / C) * 
                                           exp( -(birth_rates[child] - death_rates[child]) * (diff_time - child_diff_time) )))

                U_t += U_child_s * transition_probabilities.loc[diff_node, child]
            
            U[(diff_node, diff_time)] = U_t
    
    
    def calculate_likelihood_along_edge(u, v, t, state):
        key = (u, v, t, state)
        if key in L:
            return L[key]
    
        diff_time = diff_T.nodes.get(state, {}).get('diff_time')
        if diff_time is None: diff_time = 0
        # else: diff_time = 1 - diff_time

        v_node = cell_lineage_T.nodes[v]
        v_t = v_node['T']
        
        if v_t < diff_time < t:  # Change of state within the edge
            
            U_t = U[(state, diff_time)]
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]) + 1e-50)

            flow = exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time)) * \
                    ( (C - 1) / 
                      (C - exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time))) ) ** 2
            
            p = 0
            for child_state in diff_T.successors(state):
                p += transition_probabilities.loc[state, child_state] * calculate_likelihood_along_edge(u, v, diff_time, child_state)

            L[key] = flow * p

        else: # Edge ends with boundary condition
            
            U_t = U[(state, diff_time)]
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]))
            flow = exp( -(birth_rates[state] - death_rates[state]) * (t - v_t)) * \
                    ( (C - exp( -(birth_rates[state] - death_rates[state]) * (v_t - diff_time))) / 
                      (C - exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time))) ) ** 2

            # print(u, v, '|', C, '|', t, v_t, '=>', flow)
            
            children = list(cell_lineage_T.successors(v))
            
            if len(children) == 0:  # Boundary condition: current time 
                v_state = v_node['state']
                if state == v_state or v_state == "?": p = sampling_rate
                else: p = 0
            
            elif len(children) == 2:  # Boundary condition: birth event
                L1 = calculate_likelihood_along_edge(v, children[0], v_t, state)
                L2 = calculate_likelihood_along_edge(v, children[1], v_t, state)
                p = L1 * L2 * birth_rates[state]

            else:
                raise Exception("#Children must be 0 or 2")
            
            L[key] = flow * p
        
        return L[key]


    for parent in nx.dfs_postorder_nodes(cell_lineage_T):
        node = cell_lineage_T.nodes[parent]
        node_time = node['T']
        node_state = node.get('state')
        children = list(cell_lineage_T.successors(parent))
        
        for child in children:
            if node_state is not None and node_state != "?":
                calculate_likelihood_along_edge(parent, child, node_time, node_state)
            else:
                for state in states:
                    calculate_likelihood_along_edge(parent, child, node_time, state)
    
    root_node = cell_lineage_T.graph['root']
    root_child = next(cell_lineage_T.successors(root_node))
    diff_T_root = [n for n, d in diff_T.in_degree() if d == 0][0]
    return L[(root_node, root_child, root_time, diff_T_root)], L, U



import numpy as np
import networkx as nx
import pandas as pd
from math import exp, log
from decimal import Decimal, getcontext


def explain_failure_at_state(T, diff_T, edge, state):
    u, v = edge

    # --- collect subtree leaves ---
    leaves = []
    for n in nx.descendants(T, v):
        if T.out_degree(n) == 0:
            leaves.append((n, T.nodes[n].get("state")))

    print("\n=== FAILURE EXPLANATION ===")
    print(f"Edge: {edge}")
    print(f"State: {state}")
    print(f"Leaf states under subtree:")
    for n, s in leaves:
        print(f"  node {n}: state={s}")

    # --- reachable sets ---
    reachable = {
        s: nx.descendants(diff_T, s) | {s}
        for s in diff_T.nodes()
    }

    print("\n--- Checking each child of failing state ---")
    for child_state in diff_T.successors(state):
        allowed = reachable[child_state]

        print(f"\n{state} → {child_state}")
        print(f"  reachable: {allowed}")

        bad = []
        for n, leaf_state in leaves:
            if leaf_state not in (None, "?") and leaf_state not in allowed:
                bad.append((n, leaf_state))

        if not bad:
            print("  ✅ This branch is structurally valid (unexpected)")
        else:
            print("  ❌ Violations:")
            for n, s in bad:
                print(f"     leaf {n} has state {s} (not reachable)")


def calculate_log_likelihood_graph_diff(
    T: nx.DiGraph,
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities: pd.DataFrame,
    diff_T: nx.DiGraph,
):
    # =========================
    # Helper assertions
    # =========================
    def assert_finite(x, msg):
        if not np.isfinite(x):
            raise RuntimeError(f"{msg}: {x}")

    def assert_prob(x, msg, eps=1e-12):
        assert_finite(x, msg)
        if not (-eps <= x <= 1 + eps):
            raise RuntimeError(f"{msg}: not in [0,1], got {x}")

    # =========================
    # Input validation
    # =========================
    for s in birth_rates:
        if birth_rates[s] <= 0:
            raise RuntimeError(f"birth rate <= 0 for state {s}")
        if death_rates[s] < 0:
            raise RuntimeError(f"death rate < 0 for state {s}")
        if birth_rates[s] == death_rates[s]:
            raise RuntimeError(f"degenerate case λ=μ at state {s}")

    if not (0 <= sampling_rate <= 1):
        raise RuntimeError(f"invalid sampling rate {sampling_rate}")

    for i in transition_probabilities.index:
        row_sum = transition_probabilities.loc[i].sum()
        if abs(row_sum - 1) > 1e-6:
            raise RuntimeError(f"transition row {i} not stochastic: {row_sum}")

    # =========================
    # Compute U
    # =========================
    states = list(diff_T.nodes())
    U = dict()

    for state in states:
        U[(state, 0)] = 1 - sampling_rate
        assert_prob(U[(state, 0)], f"U init invalid at {state}")

    for diff_node in nx.dfs_postorder_nodes(diff_T):
        diff_time = diff_T.nodes[diff_node].get('diff_time') or 0

        if diff_time > 0:
            U_t = 0
            children = list(diff_T.successors(diff_node))

            for child in children:
                child_diff_time = diff_T.nodes[child].get('diff_time') or 0

                if child_diff_time > diff_time:
                    raise RuntimeError(
                        f"time violation: child {child_diff_time} > parent {diff_time}"
                    )

                U_child = U[(child, child_diff_time)]
                assert_prob(U_child, f"U_child invalid at {child}")

                lam = birth_rates[child]
                mu = death_rates[child]

                denom = (U_child - (mu / lam))
                if abs(denom) < 1e-12:
                    raise RuntimeError(
                        f"C denom ~0 at child={child}, U={U_child}, mu/lambda={mu/lam}"
                    )

                C = (U_child - 1) / denom
                assert_finite(C, f"C invalid at {child}")

                delta_t = diff_time - child_diff_time
                if delta_t < 0:
                    raise RuntimeError(f"negative time diff at {child}")

                exp_term = exp(-(lam - mu) * delta_t)
                assert_finite(exp_term, f"exp overflow at {child}")

                denom2 = (1 - (1 / C) * exp_term)
                if abs(denom2) < 1e-12:
                    raise RuntimeError(
                        f"U_child_s denom ~0 at {child}, C={C}, exp={exp_term}"
                    )

                U_child_s = 1 - (lam - mu) / (lam * denom2)
                assert_prob(U_child_s, f"U_child_s invalid at {child}")

                tp = transition_probabilities.loc[diff_node, child]
                assert_prob(tp, f"transition prob invalid {diff_node}->{child}")

                U_t += U_child_s * tp

            assert_prob(U_t, f"U_t invalid at {diff_node}")
            U[(diff_node, diff_time)] = U_t

    # =========================
    # Decimal precision
    # =========================
    ctx = getcontext()
    ctx.prec = 50
    ctx.Emax = 99999999
    ctx.Emin = -99999999

    LL = dict()

    # =========================
    # Recursive likelihood
    # =========================
    def calculate_log_likelihood_along_edge(u, v, t, state):
        key = (u, v, t, state)
        if key in LL:
            return LL[key]

        diff_time = diff_T.nodes.get(state, {}).get('diff_time') or 0
        v_t = T.nodes[v]['T']

        lam = birth_rates[state]
        mu = death_rates[state]

        U_t = U[(state, diff_time)]
        assert_prob(U_t, f"U_t invalid at state {state}")

        denom = (U_t - (mu / lam))
        if abs(denom) < 1e-12:
            raise RuntimeError(
                f"C denom ~0 at state={state}, U={U_t}, mu/lambda={mu/lam}"
            )

        C = (U_t - 1) / denom
        assert_finite(C, f"C invalid at state {state}")

        # =====================
        # Case 1: state change
        # =====================
        if v_t <= diff_time <= t:
            delta = t - diff_time
            if delta < 0:
                raise RuntimeError("negative delta")

            exp_term = exp(-(lam - mu) * delta)
            assert_finite(exp_term, "exp overflow")

            denom_flow = (C - exp_term)
            if abs(denom_flow) < 1e-12:
                raise RuntimeError(
                    f"flow denom ~0 (state change): C={C}, exp={exp_term}"
                )

            flow = exp_term * ((C - 1) / denom_flow) ** 2
            if flow <= 0:
                raise RuntimeError(f"flow <= 0: {flow}")
            assert_finite(flow, "flow invalid")

            p = Decimal(0)

            for child_state in diff_T.successors(state):
                left_ll = Decimal(
                    calculate_log_likelihood_along_edge(u, v, diff_time, child_state)
                )
                left = left_ll.exp()

                tp = Decimal(transition_probabilities.loc[state, child_state])
                if tp < 0:
                    raise RuntimeError("negative transition prob")

                p += tp * left

            if p <= 0:
                explain_failure_at_state(T, diff_T, (u, v), state)
                raise RuntimeError(f"mixture probability p=0 at state {state}")
            LL[key] = float(Decimal(flow).ln() + p.ln())

        # =====================
        # Case 2: boundary
        # =====================
        else:
            exp1 = exp(-(lam - mu) * (v_t - diff_time))
            exp2 = exp(-(lam - mu) * (t - diff_time))

            assert_finite(exp1, "exp1 invalid")
            assert_finite(exp2, "exp2 invalid")

            denom_flow = (C - exp2)
            if abs(denom_flow) < 1e-12:
                raise RuntimeError(
                    f"flow denom ~0 (boundary): C={C}, exp2={exp2}"
                )

            flow = exp(-(lam - mu) * (t - v_t)) * ((C - exp1) / denom_flow) ** 2

            if flow <= 0:
                raise RuntimeError(f"flow <= 0: {flow}")
            assert_finite(flow, "flow invalid")

            children = list(T.successors(v))

            # Leaf
            if len(children) == 0:
                v_state = T.nodes[v].get('state')

                if state == v_state or v_state == "?":
                    if sampling_rate <= 0:
                        raise RuntimeError("log(0) from sampling rate")

                    LL[key] = log(flow) + log(sampling_rate)
                else:
                    LL[key] = -np.inf

            # Birth
            elif len(children) == 2:
                LL1 = calculate_log_likelihood_along_edge(v, children[0], v_t, state)
                LL2 = calculate_log_likelihood_along_edge(v, children[1], v_t, state)

                if birth_rates[state] <= 0:
                    raise RuntimeError("invalid birth rate in log")

                LL[key] = log(flow) + LL1 + LL2 + log(birth_rates[state])

            else:
                raise RuntimeError("#Children must be 0 or 2")

        # assert_finite(LL[key], f"LL is NaN at key {key}")

        return LL[key]

    # =========================
    # Traverse tree
    # =========================
    for parent in nx.dfs_postorder_nodes(T):
        node = T.nodes[parent]
        node_time = node['T']
        node_state = node.get('state')
        children = list(T.successors(parent))

        for child in children:
            if node_state is not None and node_state != "?":
                calculate_log_likelihood_along_edge(parent, child, node_time, node_state)
            else:
                for state in states:
                    calculate_log_likelihood_along_edge(parent, child, node_time, state)

    def debug_ll_path(LL, T, diff_T, top_k=3):
        import numpy as np
        from collections import defaultdict

        # -------------------------
        # Group LL by edge
        # -------------------------
        edge_map = defaultdict(dict)
        for (u, v, t, state), val in LL.items():
            edge_map[(u, v, t)][state] = val

        def summarize_edge(u, v, t):
            states_dict = edge_map[(u, v, t)]

            items = list(states_dict.items())
            items.sort(key=lambda x: x[1], reverse=True)

            best_state, best_val = items[0]
            all_neg_inf = all(np.isneginf(val) for _, val in items)

            print(f"\nEDGE ({u} → {v}, t={t:.4f})")

            if all_neg_inf:
                print("  ❌ ALL STATES = -inf → FAILURE HERE")
            else:
                print(f"  ✅ best state: {best_state}, LL={best_val:.4f}")

            print("  top states:")
            for s, val in items[:top_k]:
                print(f"    {s}: {val}")

            return best_state, best_val, all_neg_inf

        # -------------------------
        # Find root edge
        # -------------------------
        root = [n for n, d in T.in_degree() if d == 0][0]
        child = next(T.successors(root))
        t0 = T.nodes[root]['T']

        print("==== DEBUGGING LL PATH ====")

        # -------------------------
        # Recursive descent
        # -------------------------
        def recurse(u, v, t, depth=0):
            indent = "  " * depth

            if (u, v, t) not in edge_map:
                print(indent + f"Missing edge ({u},{v},{t}) in LL")
                return

            best_state, best_val, failed = summarize_edge(u, v, t)

            if failed:
                print(indent + "STOP: all states impossible")
                return

            # descend if binary split
            children = list(T.successors(v))

            if len(children) == 0:
                print(indent + "Reached leaf")
                return

            if len(children) == 2:
                t_child = T.nodes[v]['T']
                recurse(v, children[0], t_child, depth + 1)
                recurse(v, children[1], t_child, depth + 1)

        recurse(root, child, t0)
    
    debug_ll_path(LL, T, diff_T)

    # =========================
    # Root likelihood
    # =========================
    root_node = [n for n, d in T.in_degree() if d == 0][0]
    root_child = next(T.successors(root_node))
    root_time = T.nodes[root_node]['T']
    diff_T_root = [n for n, d in diff_T.in_degree() if d == 0][0]

    result = LL[(root_node, root_child, root_time, diff_T_root)]
    assert_finite(result, "final likelihood is NaN")


    return result


import numpy as np
import networkx as nx
from math import exp, log
from scipy.special import logsumexp


def calculate_log_likelihood_graph_diff_opt(
    T: nx.DiGraph,
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities,
    diff_T: nx.DiGraph,
):
    # =========================
    # Precompute / Cache
    # =========================
    
    states = list(diff_T.nodes())
    n_states = len(states)
    
    state_index = {s: i for i, s in enumerate(states)}
    TP = transition_probabilities.values

    birth = np.array([birth_rates[s] for s in states])
    death = np.array([death_rates[s] for s in states])
    delta = birth - death
    ratio = death / birth

    diff_children = {n: list(diff_T.successors(n)) for n in states}
    diff_time_map = {n: diff_T.nodes[n].get('diff_time') or 0 for n in states}

    T_children = {n: list(T.successors(n)) for n in T.nodes}
    T_time = {n: T.nodes[n]['T'] for n in T.nodes}
    T_state = {n: T.nodes[n].get('state') for n in T.nodes}

    diff_postorder = list(nx.dfs_postorder_nodes(diff_T))
    T_postorder = list(nx.dfs_postorder_nodes(T))

    log_sampling = log(sampling_rate)

    # =========================
    # U Computation (unchanged)
    # =========================
    
    U = {}
    for state in states:
        U[(state, 0)] = 1 - sampling_rate

    for diff_node in diff_postorder:
        diff_time = diff_time_map[diff_node]

        if diff_time > 0:
            U_t = 0.0

            for child in diff_children[diff_node]:
                child_diff_time = diff_time_map[child]
                U_child = U[(child, child_diff_time)]

                idx_child = state_index[child]

                C = (U_child - 1) / (U_child - ratio[idx_child])

                exp_term = exp(-delta[idx_child] * (diff_time - child_diff_time))

                U_child_s = 1 - (delta[idx_child]) / (
                    birth[idx_child] * (1 - (1 / C) * exp_term)
                )

                tp = TP[state_index[diff_node], idx_child]
                U_t += U_child_s * tp

            U[(diff_node, diff_time)] = U_t

    # =========================
    # Log Likelihood DP
    # =========================
    
    LL = {}

    def calculate_log_likelihood_along_edge(u, v, t, state):
        key = (u, v, t, state)
        if key in LL:
            return LL[key]

        diff_time = diff_time_map.get(state, 0)
        v_t = T_time[v]
        idx_state = state_index[state]

        U_t = U[(state, diff_time)]
        C = (U_t - 1) / (U_t - ratio[idx_state] + 1e-200)

        if v_t < diff_time < t:
            dt = t - diff_time
            exp_term = exp(-delta[idx_state] * dt)

            flow = exp_term * ((C - 1) / (C - exp_term)) ** 2
            log_flow = log(flow)

            log_terms = []
            for child_state in diff_children[state]:
                idx_child = state_index[child_state]
                tp = TP[idx_state, idx_child]

                if tp == 0:
                    continue

                child_ll = calculate_log_likelihood_along_edge(
                    u, v, diff_time, child_state
                )

                log_terms.append(log(tp) + child_ll)

            if len(log_terms) == 0:
                LL[key] = -np.inf
            else:
                LL[key] = log_flow + logsumexp(log_terms)

        else:
            dt1 = t - v_t
            dt2 = v_t - diff_time
            dt3 = t - diff_time

            exp1 = exp(-delta[idx_state] * dt1)
            exp2 = exp(-delta[idx_state] * dt2)
            exp3 = exp(-delta[idx_state] * dt3)

            flow = exp1 * ((C - exp2) / (C - exp3)) ** 2
            log_flow = log(flow)

            children = T_children[v]

            if len(children) == 0:
                v_state = T_state[v]

                if state == v_state or v_state == "?":
                    LL[key] = log_flow + log_sampling
                else:
                    LL[key] = -np.inf

            elif len(children) == 2:
                LL1 = calculate_log_likelihood_along_edge(v, children[0], v_t, state)
                LL2 = calculate_log_likelihood_along_edge(v, children[1], v_t, state)

                LL[key] = log_flow + LL1 + LL2 + log(birth[idx_state])

            else:
                raise Exception("#Children must be 0 or 2")

        return LL[key]

    # =========================
    # Run DP
    # =========================
    
    for parent in T_postorder:
        node_time = T_time[parent]
        node_state = T_state[parent]

        for child in T_children[parent]:
            if node_state is not None and node_state != "?":
                calculate_log_likelihood_along_edge(
                    parent, child, node_time, node_state
                )
            else:
                for state in states:
                    calculate_log_likelihood_along_edge(
                        parent, child, node_time, state
                    )

    # =========================
    # Root extraction
    # =========================
    
    root_node = next(n for n, d in T.in_degree() if d == 0)
    root_child = T_children[root_node][0]
    root_time = T_time[root_node]

    diff_root = next(n for n, d in diff_T.in_degree() if d == 0)

    return LL[(root_node, root_child, root_time, diff_root)]




def calculate_likelihood_2(
    birth_rates: dict,
    death_rates: dict,
    transition_probabilities: pd.DataFrame,
    cell_lineage_T,
    diff_T,
    diff_T_root='A',
):
    prob = 1
    # lambdas
    internal_nodes = [n for n in cell_lineage_T.nodes 
                      if list(cell_lineage_T.successors(n)) and list(cell_lineage_T.predecessors(n))
                      ]

    for node in internal_nodes:
        # print(birth_rates[cell_lineage_T.nodes[node]['state']])
        prob *= birth_rates[cell_lineage_T.nodes[node]['state']]
    
    # edges
    # calculate end Es:
    end_Es = {}
    Cks = {}
    dfs_nodes = list(nx.dfs_postorder_nodes(diff_T, source=diff_T_root))
    for node in dfs_nodes:
        if diff_T.nodes[node]["diff_time_rev"] is None:
            end_Es[node] = 0
            Cks[node] = birth_rates[node] / death_rates[node]
        
        else:
            # calculate boundary Es for children
            children_Es = {}
            children = list(diff_T.successors(node))
            for child in children:
                numerator = (birth_rates[child] - death_rates[child])
                node_t = diff_T.nodes[node]["diff_time_rev"]
                child_t = diff_T.nodes[child].get("diff_time_rev", 0)
                if child_t is None: child_t = 0
                denominator = birth_rates[child] * (1 - 1 / Cks[child] * 
                    exp( -(birth_rates[child] - death_rates[child]) * 
                        (node_t - child_t)))
                E = 1 - numerator / denominator
                children_Es[child] = E

            total_Es = 0; total_transition_w = 0
            for child, E in children_Es.items():
                t_prb = transition_probabilities.loc[node, child]
                total_Es += E * t_prb
                total_transition_w += t_prb
            
            end_Es[node] = total_Es / total_transition_w
            Cks[node] = (end_Es[node] - 1) / (end_Es[node] - death_rates[node] / birth_rates[node])
    
    for edge in cell_lineage_T.edges:
        
        u, v = edge
        u_type = cell_lineage_T.nodes[u]["type"]
        v_type = cell_lineage_T.nodes[v]["type"]

        type_path = nx.shortest_path(diff_T, source=u_type, target=v_type)

        # back calculate Es and Cks
        fixed_Cks = {}
        fixed_end_Es = {}

        last_t = type_path[-1]

        fixed_end_Es[last_t] = end_Es[last_t]
        fixed_Cks[last_t] = (fixed_end_Es[last_t] - 1) / (fixed_end_Es[last_t] - death_rates[last_t] / birth_rates[last_t])

        for t, child in zip(type_path[-2::-1], type_path[::-1]):
            # print(curr, child)
            numerator = (birth_rates[child] - death_rates[child])
            node_t = diff_T.nodes[t]["diff_time_rev"]
            child_t = diff_T.nodes[child].get("diff_time_rev", 0)
            if child_t is None: child_t = 0
            denominator = birth_rates[child] * (1 - 1 / fixed_Cks[child] * 
                exp( -(birth_rates[child] - death_rates[child]) * 
                    (node_t - child_t)))
            E = 1 - numerator / denominator
            fixed_end_Es[t] = E
            fixed_Cks[t] = (fixed_end_Es[t] - 1) / (fixed_end_Es[t] - death_rates[t] / birth_rates[t])

        # calculate Probabilities
        start = cell_lineage_T.nodes[u]["T"]

        for t in type_path[:-1]:
            type_end = diff_T.nodes[t]['diff_time_rev']
            if type_end is None: type_end = 0

            birth_death_diff_rate = birth_rates[t] - death_rates[t]

            p = exp( -(birth_death_diff_rate) * (start - type_end)) * \
                (fixed_Cks[t] - 1) ** 2 / \
                (fixed_Cks[t] - exp( -(birth_death_diff_rate) * (start - type_end))) ** 2
            
            prob *= p
            start = type_end
    
        end = cell_lineage_T.nodes[v]["T"]
        type_end = diff_T.nodes[v_type]['diff_time_rev']
        if type_end is None: type_end = 0

        birth_death_diff_rate = birth_rates[v_type] - death_rates[v_type]
        p = exp( -(birth_death_diff_rate) * (start - end)) * \
            (Cks[v_type] - exp( -(birth_death_diff_rate) * (end - type_end))) ** 2 / \
            (Cks[v_type] - exp( -(birth_death_diff_rate) * (start - type_end))) ** 2

        # print(start, end, type_end, p)

        # print(u, v, '|', Cks[v_type], '|', start, end, '=>', p)

        prob *= p
    
        # transition probs
        for i in range(len(type_path) - 1):
            cur_t = type_path[i]
            next_t = type_path[i+1]
            p = transition_probabilities.loc[cur_t, next_t]

            prob *= p

    return prob



def assign_states_to_internal_nodes(cell_lineage_T, diff_T, correct=False):

    ## assumes both cell_lineaage_T and diff_T is reversed, i.e. time goes from leaves to root from 0 to 1.
    

    lineage_nodes = list(cell_lineage_T.nodes(data=True))
    lineage_nodes.reverse()
    diff_nodes = list(diff_T.nodes(data=True))
    
    combined_list = lineage_nodes + diff_nodes
        
    combined_list = [
        (n, d) for n, d in combined_list 
        if d.get('T') is not None or d.get('diff_time') is not None
    ]

    combined_sorted = sorted(
        combined_list,
        key=lambda x: x[1].get('T', x[1].get('diff_time'))
    )

    transform_map = {}

    for node in combined_sorted:
        node_id, node_data = node

        if 'T' in node_data:

            children = list(cell_lineage_T.successors(node_id))

            children_types = [cell_lineage_T.nodes[ch]['state'] 
                            for ch in children 
                            if 'state' in cell_lineage_T.nodes[ch]]
            
            children_types = [transform_map.get(t, t) for t in children_types]

            if 'state' in node_data: continue

            unique_types = list(set(children_types))
            if "?" in unique_types:
                unique_types.remove("?")

            if len(unique_types) > 1:
                
                if correct:
                    print(f"Children have > 1 types {unique_types} at {node_data['T']}")
                    cell_lineage_T.nodes[node_id]['state'] = "?"
                    descendants = nx.descendants(cell_lineage_T, node_id)
                    for d in descendants:
                        print(f"Setting state for descendant {d} at {cell_lineage_T.nodes[d]['T']} to ?")
                        cell_lineage_T.nodes[d]['state'] = "?"
                else:
                    raise ValueError(f"Children have > 1 types {unique_types}")

            else:
                cell_lineage_T.nodes[node_id]["state"] = unique_types[0]

        elif 'diff_time' in node_data:

            children = list(diff_T.successors(node_id))
            for child in children:
                transform_map[child] = node_id
            
            for key, value in list(transform_map.items()):
                for child in children:
                    if child == value:
                        transform_map[key] = node_id
    
    return cell_lineage_T


def add_node_times(cell_lineage_T, root=0, reverse=True):
    distances = nx.single_source_dijkstra_path_length(cell_lineage_T, root, weight='weight')
    for node, dist in distances.items():
        if reverse:
            cell_lineage_T.nodes[node]['T'] = 1 - dist
        else:
            cell_lineage_T.nodes[node]['T'] = dist
    
    return cell_lineage_T



if __name__ == "__main__":
        
    transition_prob = [
        [0, 0.5, 0.5, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
    types = ['A', 'B', 'C', 'D', 'E']
    transition_probabilities = pd.DataFrame(transition_prob, index=types, columns=types)

    calculate_likelihood_of_tree(
        birth_rates={'A': 1.2, 'B': 1.0, 'C': 1.2, 'D': 0.9, 'E': 0.8,},
        death_rates={'A': 0.2, 'B': 0.1, 'C': 0.2, 'D': 0.3, 'E': 0.3,}, 
        transition_probabilities=transition_probabilities,
        cell_lineage_T=cell_lineage_T, 
        diff_T=diff_T
    )
