import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

color_map = {
    "NMPs": "orange",
    "NeuralTube": "skyblue",
    "Somite": "green",
    "Endoderm": "yellow",
    "Endothelial": "blue",
    "PGCLC": "purple",
    "MP": "pink",
    "NMSTD": "cyan",
    "NMSD": "gold",
    "NMS": "brown",
    "NS": "lightcoral",
    "?": "red",
    # positive integers 0–15
    "0": "#f7fbff",
    "1": "#deebf7",
    "2": "#c6dbef",
    "3": "#9ecae1",
    "4": "#6baed6",
    "5": "#4292c6",
    "6": "#2171b5",
    "7": "#08519c",
    "8": "#08306b",
    "9": "#a1d99b",
    "10": "#74c476",
    "11": "#41ab5d",
    "12": "#238b45",
    "13": "#006d2c",
    "14": "#00441b",
    "15": "purple",  # existing
    # negative integers -1 to -15
    "-1": "#fee5d9",
    "-2": "#fcbba1",
    "-3": "#fc9272",
    "-4": "#fb6a4a",
    "-5": "#ef3b2c",
    "-6": "#cb181d",
    "-7": "#a50f15",
    "-8": "#67000d",
    "-9": "#fdd0a2",
    "-10": "#fdae6b",
    "-11": "#fd8d3c",
    "-12": "#f16913",
    "-13": "#d94801",
    "-14": "#8c2d04",
    "-15": "#662506",
    "K1": "blue",
    "T1": "green",
    "N1": "orange",
    0: "green",
    1: "orange",
    2: "blue",
    3: "pink",
    4: "brown",
    "A": "blue",
    "B": "green",
    "C": "brown",
}


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
        v
        for v in nx.dfs_postorder_nodes(G, source=root)
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

        state = G.nodes[v]["state"]
        color = color_map.get(state, "black")

        x = x_pos[v]

        # vertical lineage
        ax.plot([x, x], [t0, t1], color=color)

        # horizontal split at birth
        if G.nodes[v]["event"] == "birth":
            children = list(G.successors(v))
            xs = [x_pos[c] for c in children]
            ax.plot(xs, [t1] * len(xs), color=color)

    # ---------
    # Draw node labels
    # ---------
    if show_labels:
        for n in G.nodes():
            x = x_pos[n]
            y = G.nodes[n]["time"]
            typ = G.nodes[n]["state"]
            label = f"{n}\n({typ})"
            ax.text(x + 0.12, y, label, fontsize=8, ha="center", va="bottom")

    # ---------
    # Cosmetics
    # ---------
    ax.set_xlabel("Lineages")
    ax.set_ylabel("Time")
    ax.invert_yaxis()  # time flows downward (phylo convention)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return ax


def plot_differentiation_tree(
    diff_T,
    birth_rates=None,
    death_rates=None,
    transition_probabilities=None,
    show_labels=True,
    deflate_labels=False,
    figsize=(6, 5),
):
    """
    Plot differentiation tree with optional birth/death rates and transition probabilities.

    Nodes: cell types (showing τ, b, d)
    Edges: differentiation paths (showing transition probabilities)
    """
    diff_T = diff_T.copy()  # avoid modifying original graph
    plt.figure(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(diff_T, prog="dot")

    # Node labels
    node_labels = {}
    for n in diff_T.nodes:
        label = f"{n}"
        if diff_T.nodes[n].get("diff_time") is not None:
            label += f"\nτ={diff_T.nodes[n]['diff_time']:.2f}"
        if diff_T.nodes[n].get("diff_times") is not None:
            times_str = ", ".join(
                f"{t:.2f}" for t in diff_T.nodes[n]["diff_times"] if t is not None
            )
            if times_str:
                label += f"\nτs=[{times_str}]"
        if deflate_labels:
            if f"{n}_0" in birth_rates:
                label += f"\n$\\mu_b$={birth_rates.get(f'{n}_0', 0):.2f}"
            else:
                label += f"\n$\\mu_b$={birth_rates.get(n, 0):.2f}"
            if f"{n}_0" in death_rates:
                label += f"\n$\\mu_d$={death_rates.get(f'{n}_0', 0):.2f}"
            else:
                label += f"\n$\\mu_d$={death_rates.get(n, 0):.2f}"
        else:
            if birth_rates is not None:
                label += f"\n$\\mu_b$={birth_rates.get(n, 0):.2f}"
            if death_rates is not None:
                label += f"\n$\\mu_d$={death_rates.get(n, 0):.2f}"
        node_labels[n] = label

    # Edge labels (transition probabilities)
    edge_labels = {}
    if transition_probabilities is not None:
        if deflate_labels:
            for u, v in diff_T.edges:
                uc = u
                vc = v
                if f"{u}_0" in transition_probabilities.index:
                    uc = f"{u}_0"
                if f"{v}_0" in transition_probabilities.columns:
                    vc = f"{v}_0"

                prob = (
                    transition_probabilities.loc[uc, vc]
                    if (
                        uc in transition_probabilities.index
                        and vc in transition_probabilities.columns
                    )
                    else 0.0
                )
                edge_labels[(u, v)] = f"{prob:.3f}"

            for n in diff_T.nodes:
                if f"{n}_0" in transition_probabilities.index:
                    edge_labels[(n, n)] = (
                        f"{transition_probabilities.loc[f'{n}_0', f'{n}_1']:.3f}"
                    )
                    diff_T.add_edge(n, n)  # self-loop for deflated node

        else:
            for u, v in diff_T.edges:
                prob = (
                    transition_probabilities.loc[u, v]
                    if (
                        u in transition_probabilities.index
                        and v in transition_probabilities.columns
                    )
                    else 0.0
                )
                edge_labels[(u, v)] = f"{prob:.3f}"

    # Node colors
    node_colors = [
        color_map.get(n, "lightblue") if color_map else "lightblue"
        for n in diff_T.nodes()
    ]

    # Draw tree
    nx.draw(
        diff_T,
        pos,
        with_labels=False,
        node_size=900,
        node_color=node_colors,
        arrows=True,
    )

    # Draw node labels (smaller font)
    if show_labels:
        nx.draw_networkx_labels(
            diff_T,
            pos,
            labels=node_labels,
            font_size=8,  # smaller font
            font_color="black",
        )

        # Draw edge labels (red, smaller)
        if transition_probabilities is not None:
            nx.draw_networkx_edge_labels(
                diff_T, pos, edge_labels=edge_labels, font_size=7, font_color="red"
            )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_ultrametric_tree(
    T,
    root,
    diff_T=None,
    add_labels=True,
    draw_birth_event_lines=False,
    red_edges=None,
    figsize=(8, 5),
    maxtime=None,
    diff_times=None,
    transition_times_of_edges=None,
    node_colors=None,
):
    # Heights from root
    height = {}

    def dfs(n, h):
        height[n] = h
        for c in T.successors(n):
            dfs(c, h + T[n][c]["length"])

    dfs(root, 0.0)

    # X positions: leaves first
    leaves = [
        v
        for v in nx.dfs_postorder_nodes(T, source=root)
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

    # Use negative heights for plotting (root at top)
    y = {n: -height[n] for n in T.nodes}
    total_height = max(height.values())
    root_time = T.nodes[root].get("T", 1.0)

    node_id_lookup = {}
    for node in T.nodes:
        node_name = getattr(node, "name", None)
        for node_id in (node, node_name, str(node), str(node_name)):
            if node_id is not None:
                try:
                    node_id_lookup[node_id] = node
                except TypeError:
                    pass

    def resolve_node(node_id):
        try:
            if node_id in T:
                return node_id
        except TypeError:
            pass
        try:
            return node_id_lookup.get(node_id, node_id_lookup.get(str(node_id)))
        except TypeError:
            return node_id_lookup.get(str(node_id))

    state_by_node = {}
    if node_colors is not None:
        for node_id, state in node_colors:
            node = resolve_node(node_id)
            if node is not None:
                state_by_node[node] = state

    transitions_by_edge = {}
    if transition_times_of_edges is not None:
        for (
            parent_id,
            child_id,
            state_0,
            state_1,
            transition_time,
        ) in transition_times_of_edges:
            parent = resolve_node(parent_id)
            child = resolve_node(child_id)
            if parent is None or child is None or not T.has_edge(parent, child):
                continue
            transitions_by_edge[(parent, child)] = (
                state_0,
                state_1,
                float(transition_time),
            )

    color_edges_by_state = (
        node_colors is not None or transition_times_of_edges is not None
    )

    def state_color(state, default="black"):
        try:
            return color_map.get(state, color_map.get(str(state), default))
        except TypeError:
            return color_map.get(str(state), default)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw rectangular tree
    for u in T.nodes:
        children = list(T.successors(u))
        if not children:
            continue
        for c in children:
            transition = transitions_by_edge.get((u, c))
            if transition is not None:
                state_0, state_1, transition_time = transition
            else:
                state_0 = state_by_node.get(u, T.nodes[u].get("state"))
                state_1 = state_by_node.get(c, T.nodes[c].get("state"))
                transition_time = None

                if state_0 is None:
                    state_0 = state_1
                if state_1 is None:
                    state_1 = state_0

            color_0 = state_color(state_0) if color_edges_by_state else "black"
            color_1 = state_color(state_1) if color_edges_by_state else "black"
            line_alpha = 0.5 if color_edges_by_state else 0.3

            # Each horizontal arm follows its outgoing edge's source state.
            ax.plot(
                [x[u], x[c]],
                [y[u], y[u]],
                color=color_0,
                alpha=line_alpha,
            )

            color = color_0
            if red_edges and (u, c) in red_edges:
                color = "red"

            if transition_time is None or color_0 == color_1:
                ax.plot(
                    [x[c], x[c]],
                    [y[u], y[c]],
                    color=color,
                    alpha=0.5 if color_edges_by_state or color == "red" else 0.3,
                )
            else:
                transition_y = -(height[root] + root_time - transition_time)
                if min(y[u], y[c]) <= transition_y <= max(y[u], y[c]):
                    ax.plot(
                        [x[c], x[c]],
                        [y[u], transition_y],
                        color=color_0,
                        alpha=0.8,
                    )
                    ax.plot(
                        [x[c], x[c]],
                        [transition_y, y[c]],
                        color=color_1,
                        alpha=0.8,
                    )
                else:
                    ax.plot([x[c], x[c]], [y[u], y[c]], color=color, alpha=0.5)
            if add_labels:
                w = T[u][c]["length"]
                y_mid = (y[u] + y[c]) / 2
                ax.text(x[c] + 0.05, y_mid, f"{w:.2f}", va="center", fontsize=8)

    # Draw nodes
    for n in T.nodes:
        state = state_by_node.get(n, T.nodes[n].get("state"))
        color = state_color(state)
        ax.scatter(x[n], y[n], color=color, zorder=3, s=5, alpha=0.3)
        node_str = str(n)
        if add_labels and state:
            node_str += "\n(" + state + ")"
        if add_labels:
            ax.text(x[n], y[n] + 0.02, node_str, ha="center", fontsize=9)

    # Draw differentiation times if provided
    if diff_T is not None:
        for n in diff_T.nodes:
            dt = diff_T.nodes[n].get("diff_time")
            if dt is not None:
                ax.axhline(-dt, zorder=10)
                children = list(diff_T.successors(n))
                if children:
                    label = f"{n}→ {{{', '.join(map(str, children))}}}"
                    ax.text(-0.15, -dt + 0.01, label, fontsize=9, zorder=11)

    if diff_times is not None:
        for dt in diff_times:
            ax.axhline(-dt, color="purple", linestyle="dotted", zorder=10)
            ax.text(
                -0.15, -dt + 0.01, f"τ={dt:.2f}", fontsize=9, color="purple", zorder=11
            )

    # Optional horizontal lines at birth events
    if draw_birth_event_lines:
        birth_nodes = [n for n in T.nodes if T.out_degree(n) > 1]
        x_min, x_max = min(x.values()), max(x.values())
        for n in birth_nodes:
            ax.hlines(
                y[n],
                xmin=x_min - 0.5,
                xmax=x_max + 0.5,
                colors="red",
                linestyles="dashed",
                alpha=0.6,
                linewidth=1.2,
            )

    # Set y-axis ticks properly
    if maxtime is None:
        maxtime = max(height.values())
    major_ticks = np.linspace(0, maxtime, 21)
    ax.set_yticks(-major_ticks)  # negative to match plotted y
    ax.set_yticklabels([f"{t:.2f}" for t in major_ticks], fontsize=8)
    ax.set_ylabel("Time")

    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.show()
