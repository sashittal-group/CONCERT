import random
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph


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
        if event == "death":
            if parent is not None:
                H.remove_edge(parent, v)
            H.remove_node(v)
            continue

        # ---------- rule 2 & 3: birth events ----------
        if event == "birth":
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




if __name__ == "__main__":
    from src.plotters import plot_simulated_lineage_tree

    save_dir = "data/simulation/test"

    G, root = simulate_constant_bd(
        lam=5.0,
        mu=.5,
        time_length=1.0,
        seed=9
    )

    for u, v, data in G.edges(data=True):
        print((u, v), data)
    
    save_tree(G, root, f"{save_dir}/T.json")
    plot_simulated_lineage_tree(G, root)

    plt.savefig(f"{save_dir}/T.png")
    plt.close()

    G_extant = filter_extant_lineages(G, root)

    save_tree(G_extant, root, f"{save_dir}/T_extant.json")

    plot_simulated_lineage_tree(G_extant, root)

    plt.savefig(f"{save_dir}/T_extant.png")
    plt.close()
