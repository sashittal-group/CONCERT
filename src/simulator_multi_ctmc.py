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


def simulate_multitype_bd_ctmc(
    birth_rates, death_rates, sampling_rates, Q, T, root_state="A", seed=None
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

    while alive and t < T:

        lineage_rates = []
        total_rate = 0.0

        for parent, tip, state in alive:

            q_out = float(Q.loc[state].sum())

            r = birth_rates[state] + death_rates[state] + q_out

            lineage_rates.append(r)
            total_rate += r

        if total_rate == 0:
            break

        dt = np.random.exponential(1.0 / total_rate)

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
            d: float(Q.loc[state, d]) for d in Q.columns if float(Q.loc[state, d]) > 0
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

        for rate, label in zip(event_rates, event_labels):
            cumulative += rate

            if u <= cumulative:
                chosen_event = label
                break

        # close current branch

        G.nodes[tip]["time"] = t
        G.edges[parent, tip]["end_time"] = t
        G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

        if chosen_event == "birth":

            G.nodes[tip]["event"] = "birth"

            for _ in range(2):

                c = next_node
                next_node += 1

                G.add_node(c, time=None, event=None, state=state)

                G.add_edge(tip, c, start_time=t, end_time=None, length=None)

                alive.append((tip, c, state))

        elif chosen_event == "death":

            G.nodes[tip]["event"] = "death"

        else:

            _, new_state = chosen_event

            G.nodes[tip]["event"] = "transition"

            c = next_node
            next_node += 1

            G.add_node(c, time=None, event=None, state=new_state)

            G.add_edge(tip, c, start_time=t, end_time=None, length=None)

            alive.append((tip, c, new_state))

    # truncate surviving branches

    for parent, tip, state in alive:

        G.nodes[tip]["time"] = T

        G.edges[parent, tip]["end_time"] = T
        G.edges[parent, tip]["length"] = T - G.edges[parent, tip]["start_time"]

        if np.random.random() < sampling_rates[state]:
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
                    (G.nodes[current]["time"], G.nodes[p]["state"], G.nodes[c]["state"])
                )

            children = list(G.successors(current))

            if len(children) == 0:
                break

            current = children[0]

        return current, transitions, start_time

    for u in keep:

        for child in G.successors(u):

            target, transitions, start_time = walk_path(u, child)

            end_time = G.nodes[target]["time"]

            H.add_edge(
                u,
                target,
                start_time=start_time,
                end_time=end_time,
                length=end_time - start_time,
                transitions=transitions,
            )

            for time, s1, s2 in transitions:

                transitions_list.append((time, u, target, s1, s2))

    transitions_list.sort(key=lambda x: x[0])

    return H, transitions_list



def save_tree(G, root, filename):
    """
    Save a birth–death tree with all node/edge attributes to a JSON file.
    """

    data = {"root": root, "graph": json_graph.node_link_data(G)}

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
                sid = ord(state) - ord("A")

            # print(n, data)
            writer.writerow([n, float(data["T"]), int(is_leaf), sid])

    # --- write edges ---
    with open(edge_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "branch_length"])

        for u, v, data in T.edges(data=True):
            writer.writerow([u, v, float(data.get("length", 0.0))])



def main():
    if __package__:
        from .plotters import plot_simulated_lineage_tree
        from .likelihood import add_node_times
    else:
        from plotters import plot_simulated_lineage_tree
        from likelihood import add_node_times

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(
        repo_root,
        "data",
        "simulation",
        "bdt_ctmc",
        "emperical_proof_data",
    )
    os.makedirs(output_dir, exist_ok=True)

    birth_rates = {"A": 7, "B": 6}
    death_rates = {"A": 1.5, "B": 0.5}
    sampling_rates = {"A": 0.5, "B": 0.4}
    Q = pd.DataFrame(
        [[0.0, 1.0], [0.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    tree_count = 100
    time_horizon = 1.0
    saved_count = 0
    seed = 0

    while saved_count < tree_count:
        print(seed)
        if seed >= 1000:
            raise RuntimeError(
                f"Generated only {saved_count} sampled trees after 1000 attempts"
            )

        graph, root = simulate_multitype_bd_ctmc(
            birth_rates=birth_rates,
            death_rates=death_rates,
            sampling_rates=sampling_rates,
            Q=Q,
            T=time_horizon,
            root_state="A",
            seed=seed,
        )
        seed += 1

        graph = add_node_times(graph, root=root)
        
        extant_tree = filter_extant_lineages(graph, root)
        if not any(
            extant_tree.nodes[node].get("event") == "sample"
            for node in extant_tree
        ):
            continue

        exported_tree, transitions = contract_edges(extant_tree, root)

        tree_dir = os.path.join(output_dir, f"tree_{saved_count + 1:03d}")
        os.makedirs(tree_dir, exist_ok=True)
        export_tree(
            exported_tree,
            node_file=os.path.join(tree_dir, "nodes.csv"),
            edge_file=os.path.join(tree_dir, "edges.csv"),
        )

        plot_simulated_lineage_tree(graph, root, show_labels=False)
        plt.savefig(os.path.join(tree_dir, "T.png"))
        plt.close()

        plot_simulated_lineage_tree(exported_tree, root, show_labels=False)
        plt.savefig(os.path.join(tree_dir, "T_extant.png"))
        plt.close()

        ordered_transitions = sorted(transitions)
        with open(os.path.join(tree_dir, "transition_times.csv"), "w") as file:
            print(len(ordered_transitions), file=file)
            for event_time, _, _, _, _ in ordered_transitions:
                print(event_time, file=file)

        with open(
            os.path.join(tree_dir, "transition_times_with_edges.csv"),
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow([len(ordered_transitions)])
            writer.writerows(ordered_transitions)

        saved_count += 1

    print(
        f"Exported {saved_count} sampled trees to {output_dir} "
        f"using {seed} simulation attempts."
    )


if __name__ == "__main__":
    main()
