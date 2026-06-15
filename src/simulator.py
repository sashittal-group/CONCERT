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
    lam,
    mu,
    Qs,
    diff_T,
    time_length,
    root_type="A",
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.DiGraph()
    next_node = 0

    root = next_node
    next_node += 1
    G.add_node(root, time=0.0, event="root", type=root_type)

    child = next_node
    next_node += 1
    G.add_node(child, time=None, event=None, type=root_type)
    G.add_edge(root, child, start_time=0.0, end_time=None, length=None)

    alive = [(root, child, root_type)]

    t = 0.0

    while t < time_length and alive:

        # ---- compute BD total rate ----
        rates = []
        total_rate = 0.0

        for parent, tip, typ in alive:
            r = lam[typ] + mu[typ]
            rates.append(r)
            total_rate += r

        # BD event time
        if total_rate > 0:
            dt_bd = random.expovariate(total_rate)
            t_bd = t + dt_bd
        else:
            t_bd = float("inf")

        # ---- compute next differentiation time ----
        t_diff = float("inf")
        diff_index = None

        for i, (parent, tip, typ) in enumerate(alive):
            dt = diff_T.nodes[typ]["diff_time"]
            if dt is not None and dt > t and dt < t_diff:
                t_diff = dt

        # ---- determine next event ----
        if t_bd < t_diff and t_bd < time_length:
            # ===== Birth or Death =====
            t = t_bd

            r = random.random() * total_rate
            cumulative = 0
            for i, rate in enumerate(rates):
                cumulative += rate
                if r <= cumulative:
                    chosen_index = i
                    break

            parent, tip, typ = alive.pop(chosen_index)

            G.nodes[tip]["time"] = t
            G.edges[parent, tip]["end_time"] = t
            G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

            if random.random() < lam[typ] / (lam[typ] + mu[typ]):
                # birth
                G.nodes[tip]["event"] = "birth"

                for _ in range(2):
                    c = next_node
                    next_node += 1
                    G.add_node(c, time=None, event=None, type=typ)
                    G.add_edge(tip, c, start_time=t, end_time=None, length=None)
                    alive.append((tip, c, typ))
            else:
                # death
                G.nodes[tip]["event"] = "death"

        elif t_diff < time_length:
            # ===== Differentiation =====
            t = t_diff

            new_alive = []

            for parent, tip, typ in alive:
                dt = diff_T.nodes[typ]["diff_time"]

                if dt == t_diff:
                    # close current edge
                    G.nodes[tip]["time"] = t
                    G.nodes[tip]["event"] = "diff"

                    G.edges[parent, tip]["end_time"] = t
                    G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

                    children = list(diff_T.successors(typ))

                    if children:
                        # Extract probabilities for allowed children
                        probs = Qs.loc[typ, children].values.astype(float)

                        # Optional safety normalization (recommended)
                        total = probs.sum()
                        if total <= 0:
                            raise ValueError(f"No positive transition probability from {typ}")
                        probs = probs / total

                        # Sample according to probability vector
                        new_type = np.random.choice(children, p=probs)

                        c = next_node
                        next_node += 1
                        G.add_node(c, time=None, event=None, type=new_type)
                        G.add_edge(tip, c, start_time=t, end_time=None, length=None)

                        new_alive.append((tip, c, new_type))
                else:
                    new_alive.append((parent, tip, typ))

            alive = new_alive

        else:
            break

    # ---- truncate survivors ----
    for parent, tip, typ in alive:
        G.nodes[tip]["time"] = time_length
        G.nodes[tip]["event"] = "sample"
        G.edges[parent, tip]["end_time"] = time_length
        G.edges[parent, tip]["length"] = (
            time_length - G.edges[parent, tip]["start_time"]
        )

    return G, root


def simulate_multitype_bd_ctmc(
    birth_rates,
    death_rates,
    sampling_rates,
    Q,
    T,
    root_type="A",
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
        type=root_type
    )

    child = next_node
    next_node += 1

    G.add_node(
        child,
        time=None,
        event=None,
        type=root_type
    )

    G.add_edge(
        root,
        child,
        start_time=0.0,
        end_time=None,
        length=None
    )

    alive = [(root, child, root_type)]

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

        G.edges[parent, tip]["length"] = (
            t
            - G.edges[parent, tip]["start_time"]
        )

        if chosen_event == "birth":

            G.nodes[tip]["event"] = "birth"

            for _ in range(2):

                c = next_node
                next_node += 1

                G.add_node(
                    c,
                    time=None,
                    event=None,
                    type=state
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
                type=new_state
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
        if event in {"death", "unsampled"}:
            if parent is not None:
                H.remove_edge(parent, v)
            H.remove_node(v)
            continue

        # ---------- rule 2 & 3: birth events ----------
        if event == "birth" or event == "diff":
            children = list(H.successors(v))

            # no surviving children → delete birth
            if len(children) == 0:
                if parent is not None:
                    H.remove_edge(parent, v)
                H.remove_node(v)

            # exactly one surviving child → contract
            elif len(children) == 1:
                c = children[0]

                if parent is not None:
                    # original edges
                    e1 = H.edges[parent, v]
                    e2 = H.edges[v, c]

                    # create contracted edge
                    H.add_edge(
                        parent,
                        c,
                        start_time=e1["start_time"],
                        end_time=e2["end_time"],
                        length=e2["end_time"] - e1["start_time"]
                    )

                    H.remove_edge(parent, v)

                H.remove_edge(v, c)
                H.remove_node(v)

            # two children → keep as is
            elif len(children) == 2:
                pass

            else:
                raise RuntimeError("Birth node with >2 children")

    return H


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
            state = data.get("type")
            is_leaf = data.get("event") == "sample"

            # hard mapping
            if not is_leaf or state is None or state == "?":
                sid = -1
            else:
                sid = ord(state) - ord('A')

            writer.writerow([
                n,
                float(data["time"]),
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
                float(data.get("weight", 0.0))
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
        lam=lam,
        mu=mu,
        diff_T=diff_T,
        Qs=Qs,
        time_length=1.0,
        root_type="A",
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



def main():
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
        root_type="A",
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



if __name__ == "__main__":
    main()
