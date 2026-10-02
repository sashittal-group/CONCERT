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


def simulate_multitype_bd_with_diff(
    birth_rates,
    death_rates,
    transition_probabilities,
    sampling_rates,
    diff_T,
    time_length,
    root_state="A",
    seed=None,
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

        for parent, tip, typ in alive:
            r = birth_rates[typ] + death_rates[typ]
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

        for parent, tip, typ in alive:
            for dt in diff_T.nodes[typ]["diff_times"]:
                if dt > t and dt < t_diff:
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

            if random.random() < birth_rates[typ] / (
                birth_rates[typ] + death_rates[typ]
            ):
                # birth
                G.nodes[tip]["event"] = "birth"

                for _ in range(2):
                    c = next_node
                    next_node += 1
                    G.add_node(c, time=None, event=None, state=typ)
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

                diff_times = diff_T.nodes[typ]["diff_times"]

                if any(np.isclose(dt, t_diff) for dt in diff_times):

                    # close current edge
                    G.nodes[tip]["time"] = t
                    G.nodes[tip]["event"] = "diff"

                    G.edges[parent, tip]["end_time"] = t
                    G.edges[parent, tip]["length"] = (
                        t - G.edges[parent, tip]["start_time"]
                    )

                    children = list(diff_T.successors(typ))

                    if children:

                        child_probs = transition_probabilities.loc[
                            typ, children
                        ].values.astype(float)

                        p_stay = 1.0 - child_probs.sum()

                        if p_stay < -1e-12:
                            raise ValueError(
                                f"Transition probabilities from {typ} sum to > 1"
                            )

                        p_stay = max(0.0, p_stay)

                        states = [typ] + children
                        probs = np.concatenate([[p_stay], child_probs])
                        probs /= probs.sum()

                        new_type = np.random.choice(states, p=probs)

                        c = next_node
                        next_node += 1

                        G.add_node(c, time=None, event=None, state=new_type)

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
        G.edges[parent, tip]["end_time"] = time_length
        G.edges[parent, tip]["length"] = (
            time_length - G.edges[parent, tip]["start_time"]
        )

        if np.random.random() < sampling_rates[typ]:
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
                        length=e2["end_time"] - e1["start_time"],
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

    data = {"root": root, "graph": json_graph.node_link_data(G)}

    print(filename, root)

    with open(filename, "w") as f:
        json.dump(data, fp=f, indent=2)


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


def main2():
    from src.plotters import plot_simulated_lineage_tree

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=int, help="seed [0]", default=0)
    parser.add_argument("-o", type=str, help="output prefix", default="sample")

    args = parser.parse_args(None if sys.argv[1:] else ["-h"])

    save_dir = args.o
    seed = args.s

    diff_map_edges = [
        ("A", "B"),
        ("A", "C"),
    ]

    diff_times = [0.5, None, None]

    diff_T = nx.DiGraph()

    for i, t in enumerate(diff_times):
        diff_T.add_node(string.ascii_uppercase[i], diff_time=t)

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
        diff_T=diff_T,
        transition_probabilities=Qs,
        time_length=1.0,
        root_state="A",
        seed=seed,
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


def main5():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    num_setting = 0

    with open(
        f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/settings.csv",
        "w",
    ) as f:

        print(
            "setting,birth(A),birth(B),birth(C),transition(A>B),transition(B>C)",
            file=f,
            flush=True,
        )

        for ba in [4, 5, 6, 7, 8]:
            for bb in [4, 5, 6, 7, 8]:
                for bc in [4, 5, 6, 7, 8]:
                    for abt in [1, 2, 4]:
                        for bct in [1, 2, 4]:
                            print(
                                num_setting,
                                ba,
                                bb,
                                bc,
                                abt,
                                bct,
                                file=f,
                                flush=True,
                                sep=",",
                            )
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
    parser.add_argument("--setting", type=int)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args(None if sys.argv[1:] else ["-h"])

    setting = args.setting
    seed = args.seed

    df = pd.read_csv(
        "/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/settings.csv",
        index_col=0,
    )

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
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        ),
        transition_windows={
            ("A", "B"): [(0, 1)],
            ("B", "C"): [(0, 1)],
        },
        T=1.0,
        root_state="A",
        seed=seed,
    )

    save_dir = (
        f"/Users/samin/workspace/phylodynamic-carta/data/simulation/bdt_ctmc/{setting}"
    )

    os.makedirs(save_dir, exist_ok=True)

    # print(G.nodes(data=True))

    G = add_node_times(G)

    # save_tree(G, root, f"{save_dir}/T_{seed}.json")
    G_extant = filter_extant_lineages(G, root)

    # save_tree(G_extant, root, f"{save_dir}/T_extant_{seed}.json")

    export_tree(
        G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv"
    )

    try:
        plot_simulated_lineage_tree(G, root)
        plt.savefig(f"{save_dir}/T_{seed}.png")
        plt.close()

        plot_simulated_lineage_tree(G_extant, root)
        plt.savefig(f"{save_dir}/T_extant_{seed}.png")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


def main():
    from src.plotters import plot_simulated_lineage_tree
    from src.likelihood import add_node_times

    seed = 1
    save_dir = "data/simulation/coord_diff"

    diff_map_edges = [
        ("A", "B"),
        ("B", "C"),
    ]

    diff_times = [[0.3, 0.7], [0.5, 0.9], []]

    diff_T = nx.DiGraph()

    for i, t in enumerate(diff_times):
        diff_T.add_node(string.ascii_uppercase[i], diff_times=t)

    for s, e in diff_map_edges:
        diff_T.add_edge(s, e)

    lam = {
        "A": 7.0,
        "B": 6.0,
        "C": 5.0,
    }

    mu = {
        "A": 0,
        "B": 0,
        "C": 0,
    }

    sampling_rates = {
        "A": 0.5,
        "B": 0.5,
        "C": 0.5,
    }

    types = list(diff_T.nodes)

    Qs = pd.DataFrame(0.0, index=types, columns=types)

    Qs.loc["A", "B"] = 0.8
    Qs.loc["B", "C"] = 0.7

    G, root = simulate_multitype_bd_with_diff(
        birth_rates=lam,
        death_rates=mu,
        sampling_rates=sampling_rates,
        diff_T=diff_T,
        transition_probabilities=Qs,
        time_length=1.0,
        root_state="A",
        seed=seed,
    )

    os.makedirs(save_dir, exist_ok=True)

    G = add_node_times(G)

    save_tree(G, root, f"{save_dir}/T.json")
    G_extant = filter_extant_lineages(G, root)

    print(G_extant.nodes(data=True))

    save_tree(G_extant, root, f"{save_dir}/T_extant.json")

    export_tree(
        G_extant, f"{save_dir}/nodes_{seed}.csv", f"{save_dir}/edges_{seed}.csv"
    )

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
