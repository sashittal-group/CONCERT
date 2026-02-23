import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_simulated_lineage_tree(G, root, ax=None, show_labels=True):
    """
    Plot an edge-based birth–death tree.
    Edges are lineages, nodes are events.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # ---------
    # Find leaves (tips at final time)
    # ---------
    leaves = [
        v for v in nx.dfs_postorder_nodes(G, source=root)
        if G.out_degree(v) == 0 and v != root
    ]

    # assign x positions to leaves
    x_pos = {leaf: i for i, leaf in enumerate(leaves)}
    next_x = len(leaves)

    # ---------
    # Postorder traversal to assign x to internal nodes
    # ---------
    def assign_x(v):
        nonlocal next_x

        children = list(G.successors(v))

        if len(children) == 0:
            return x_pos[v]

        xs = [assign_x(c) for c in children]
        x_pos[v] = sum(xs) / len(xs)
        return x_pos[v]

    assign_x(root)

    # ---------
    # Draw edges
    # ---------
    for u, v, data in G.edges(data=True):
        t0 = data["start_time"]
        t1 = data["end_time"]

        x = x_pos[v]

        # vertical lineage
        ax.plot([x, x], [t0, t1], color="black")

        # horizontal split at birth
        if G.nodes[v]["event"] == "birth":
            children = list(G.successors(v))
            xs = [x_pos[c] for c in children]
            ax.plot(xs, [t1, t1], color="black")

    # ---------
    # Draw node labels
    # ---------
    if show_labels:
        for n in G.nodes():
            x = x_pos[n]
            y = G.nodes[n]["time"]
            typ = G.nodes[n]["type"]
            label = f"{n}\n({typ})"
            ax.text(x + 0.12, y, label, fontsize=8, ha="center", va="bottom")

    # ---------
    # Cosmetics
    # ---------
    ax.set_xlabel("Lineages")
    ax.set_ylabel("Time")
    ax.invert_yaxis()   # time flows downward (phylo convention)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return ax


def plot_differentiation_tree(diff_T, show_labels=True):
    """
    Plot differentiation tree.
    Nodes are cell types, edges are differentiation paths.
    """
    pos = nx.nx_agraph.graphviz_layout(diff_T, prog="dot")

    node_labels = {
        n: f"{n}\nt={diff_T.nodes[n]['diff_time']}"
        for n in diff_T.nodes
    }

    plt.figure(figsize=(5, 4))

    nx.draw(
        diff_T,
        pos,
        with_labels=False,
        node_size=900,
        node_color="lightblue",
        arrows=True
    )

    # Node labels (ID + time)
    nx.draw_networkx_labels(
        diff_T,
        pos,
        labels=node_labels
    )


def draw_ultrametric_tree(T, root, diff_T=None, draw_birth_event_lines=False):
    
    # Heights from root
    height = {}

    def dfs(n, h):
        height[n] = h
        for c in T.successors(n):
            dfs(c, h + T[n][c]["weight"])

    dfs(root, 0.0)

    # X positions: leaves first
    leaves = [
        v for v in nx.dfs_postorder_nodes(T, source=root)
        if T.out_degree(v) == 0 and v != root
    ]
    x = {leaf: i for i, leaf in enumerate(leaves)}

    def assign_x(n):
        if n in x:
            return x[n]
        children = list(T.successors(n))
        x[n] = sum(assign_x(c) for c in children) / len(children)
        return x[n]

    assign_x(root)

    y = {n: -height[n] for n in T.nodes}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw rectangular structure
    for u in T.nodes:
        children = list(T.successors(u))
        if not children:
            continue

        xs = [x[c] for c in children]

        # horizontal bar at parent height
        ax.plot([min(xs), max(xs)], [y[u], y[u]], color="black")

        # vertical branches + weights
        for c in children:
            ax.plot([x[c], x[c]], [y[u], y[c]], color="black")

            # edge weight label (midpoint of vertical segment)
            w = T[u][c]["weight"]
            y_mid = (y[u] + y[c]) / 2
            ax.text(
                x[c] + 0.05,
                y_mid,
                f"{w:.2f}",
                va="center",
                fontsize=8,
            )

    # Draw nodes
    for n in T.nodes:
        ax.scatter(x[n], y[n], color="black", zorder=3)
        node_str = str(n)
        if "type" in T.nodes[n]:
            node_str += "\n(" + (T.nodes[n]["type"]) + ")"
        ax.text(x[n] - 0.07, y[n] + 0.02, node_str, ha="center", fontsize=9)

    if diff_T is not None:
        for n in diff_T.nodes:
            dt = diff_T.nodes[n].get("diff_time")
            if dt is not None:
                ax.axhline(-dt)

                children = list(diff_T.successors(n))
                if not children: continue
                label = str(n) + "→ {" +  ', '.join(map(str, children)) + "}"
                ax.text(
                    -0.15,
                    -dt + 0.01,
                    label,
                    fontsize=9
                )
    
    if draw_birth_event_lines:
        # ---- Horizontal lines at birth events (internal nodes) ----
        birth_nodes = [n for n in T.nodes if T.out_degree(n) > 1]

        x_min = min(x.values())
        x_max = max(x.values())

        for n in birth_nodes:
            ax.hlines(
                y[n],
                xmin=x_min - 0.5,
                xmax=x_max + 0.5,
                colors="red",
                linestyles="dashed",
                alpha=0.6,
                linewidth=1.2
            )

    # ax.set_ylim(-1.0, 0.0)

    major_ticks = np.linspace(0, 1, 21)
    ax.set_yticks(major_ticks - 1)
    ax.set_yticklabels([f"{t:.2f}" for t in major_ticks], fontsize=8)

    # ax.set_ylabel("Time")

    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    plt.show()
