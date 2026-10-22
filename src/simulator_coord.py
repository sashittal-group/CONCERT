import argparse
import csv
import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph


def simulate_multitype_bd_with_diff(
    birth_rates: dict[str, float],
    death_rates: dict[str, float],
    sampling_rates: dict[str, float],
    transition_probabilities: pd.DataFrame,
    differentiation_times: nx.DiGraph,
    time_length: float,
    root_state: str = "A",
    seed: int | None = None,
):
    if not math.isfinite(time_length) or time_length <= 0:
        raise ValueError("time_length must be a finite positive number")

    states = set(differentiation_times.nodes)
    if root_state not in states:
        raise ValueError(f"Unknown root state: {root_state}")
    for state in states:
        if state not in birth_rates or state not in death_rates:
            raise ValueError(f"Missing birth/death rate for state {state!r}")
        if state not in sampling_rates:
            raise ValueError(f"Missing sampling rate for state {state!r}")
        birth_rate = float(birth_rates[state])
        death_rate = float(death_rates[state])
        sampling_rate = float(sampling_rates[state])
        if not math.isfinite(birth_rate) or birth_rate < 0:
            raise ValueError(f"Birth rate for {state!r} must be finite and non-negative")
        if not math.isfinite(death_rate) or death_rate < 0:
            raise ValueError(f"Death rate for {state!r} must be finite and non-negative")
        if not math.isfinite(sampling_rate) or not 0 <= sampling_rate <= 1:
            raise ValueError(f"Sampling rate for {state!r} must be in [0, 1]")
        if (
            state not in transition_probabilities.index
            or state not in transition_probabilities.columns
        ):
            raise ValueError(
                f"Transition probabilities must include state {state!r}"
            )

        transition_times = differentiation_times.nodes[state].get(
            "transition_times",
            [],
        )
        if any(
            not math.isfinite(float(transition_time))
            or float(transition_time) < 0
            or float(transition_time) > time_length
            for transition_time in transition_times
        ):
            raise ValueError(
                f"Transition times for {state!r} must be finite and within "
                "the simulation horizon"
            )

    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)

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
            dt_bd = python_rng.expovariate(total_rate)
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

            chosen_index = python_rng.choices(
                range(len(alive)),
                weights=rates,
                k=1,
            )[0]

            parent, tip, state = alive.pop(chosen_index)

            G.nodes[tip]["time"] = t
            G.edges[parent, tip]["end_time"] = t
            G.edges[parent, tip]["length"] = t - G.edges[parent, tip]["start_time"]

            if python_rng.random() < birth_rates[state] / (
                birth_rates[state] + death_rates[state]
            ):
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

            alive, next_node = _apply_scheduled_transitions(
                graph=G,
                alive=alive,
                differentiation_times=differentiation_times,
                transition_probabilities=transition_probabilities,
                transition_time=t,
                next_node=next_node,
                rng=numpy_rng,
            )

        else:
            break

    # ---- truncate survivors ----

    for parent, tip, state in alive:

        edge = G.edges[parent, tip]
        edge["end_time"] = time_length
        edge["length"] = time_length - edge["start_time"]
        edge["end_state"] = state
        G.nodes[tip]["time"] = time_length

        if numpy_rng.random() < sampling_rates[state]:
            G.nodes[tip]["event"] = "sample"
        else:
            G.nodes[tip]["event"] = "unsampled"

    return G, root


def _apply_scheduled_transitions(
    graph: nx.DiGraph,
    alive: list[tuple[int, int, str]],
    differentiation_times: nx.DiGraph,
    transition_probabilities: pd.DataFrame,
    transition_time: float,
    next_node: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, str]], int]:
    next_alive = []
    for parent, tip, state in alive:
        scheduled_times = differentiation_times.nodes[state].get(
            "transition_times",
            [],
        )
        if transition_time not in scheduled_times:
            next_alive.append((parent, tip, state))
            continue

        graph.nodes[tip]["time"] = transition_time
        graph.nodes[tip]["event"] = "transition"
        edge = graph.edges[parent, tip]
        edge["end_time"] = transition_time
        edge["length"] = transition_time - edge["start_time"]

        next_states = [*differentiation_times.successors(state), state]
        probabilities = transition_probabilities.loc[
            state,
            next_states,
        ].to_numpy(dtype=float)
        probability_total = float(probabilities.sum())
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or probability_total <= 0
        ):
            raise ValueError(f"Invalid transition probabilities from {state!r}")
        probabilities /= probability_total
        new_state = str(rng.choice(next_states, p=probabilities))

        child = next_node
        next_node += 1
        graph.add_node(child, time=None, event=None, state=new_state)
        graph.add_edge(
            tip,
            child,
            start_time=transition_time,
            end_time=None,
            length=None,
        )
        next_alive.append((tip, child, new_state))

    return next_alive, next_node



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

        for start, end in transition_windows.get((state, dest), []):
            if start <= t < end:
                active = True
                break

        if active:
            transitions[dest] = q

    return transitions


def filter_extant_lineages(G, root):
    """
    Prune tree bottom-up to keep only extant lineages.
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

    keep = []

    for v in G.nodes:

        children = list(G.successors(v))

        if v == root:
            keep.append(v)

        elif len(children) != 1:
            keep.append(v)

    keep_set = set(keep)

    # copy kept nodes
    for v in keep:
        H.add_node(v, **G.nodes[v])

    transitions_list = []

    def walk_path(parent_keep, child):

        current = child

        transitions = []

        start_time = G.edges[parent_keep, child]["start_time"]

        while current not in keep_set:

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


def export_tree(
    T,
    node_file="nodes.csv",
    edge_file="edges.csv",
    state_to_id: dict[str, int] | None = None,
):
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
            elif state_to_id is not None:
                if state not in state_to_id:
                    raise ValueError(f"No state ID configured for {state!r}")
                sid = state_to_id[state]
            else:
                if not isinstance(state, str) or len(state) != 1:
                    raise ValueError(
                        "state_to_id is required for non-single-character states"
                    )
                sid = ord(state) - ord("A")

            # print(n, data)
            writer.writerow([n, float(data["T"]), int(is_leaf), sid])

    # --- write edges ---
    with open(edge_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent", "child", "branch_length"])

        for u, v, data in T.edges(data=True):
            writer.writerow([u, v, float(data.get("length", 0.0))])



def read_simulation_settings(setting_file):
    """Read the section-based settings format used by setting.txt."""
    sections = {
        "num states": [],
        "states": [],
        "birth rates": [],
        "death rates": [],
        "sampling rates": [],
        "transition times": [],
        "transition probabilities": [],
        "simulation time horizon": [],
        "initial state": [],
        "num trees": [],
    }
    current_section = None

    with open(setting_file, "r") as settings:
        for line_number, raw_line in enumerate(settings, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                heading = stripped[1:].strip().lower()
                if heading in sections:
                    current_section = heading
                elif heading.startswith("transition probabilities"):
                    current_section = "transition probabilities"
                elif heading.startswith("transition times"):
                    current_section = "transition times"
                continue

            content, _, inline_heading = stripped.partition("#")
            inline_heading = inline_heading.strip().lower()
            if inline_heading in sections:
                current_section = inline_heading
            elif inline_heading.startswith("transition probabilities"):
                current_section = "transition probabilities"
            elif inline_heading.startswith("transition times"):
                current_section = "transition times"
            content = content.strip()
            if not content:
                continue
            if current_section is None:
                raise ValueError(f"Missing section heading on line {line_number}")
            sections[current_section].append((line_number, content.split()))

    def one_value(section):
        rows = sections[section]
        if len(rows) != 1 or len(rows[0][1]) != 1:
            raise ValueError(f"Expected one value in '{section}' section")
        return rows[0][1][0]

    try:
        num_states = int(one_value("num states"))
    except ValueError as error:
        raise ValueError("'num states' must contain an integer") from error

    state_rows = sections["states"]
    if len(state_rows) != 1:
        raise ValueError("'states' section must contain one list of state names")
    states = state_rows[0][1]
    if num_states != len(states) or len(set(states)) != len(states):
        raise ValueError("'num states' must match the unique names in 'states'")

    def parse_rates(section):
        values = {}
        for line_number, row in sections[section]:
            if len(row) != 2 or row[0] not in states:
                raise ValueError(f"Invalid {section} entry on line {line_number}")
            if row[0] in values:
                raise ValueError(
                    f"Duplicate state in '{section}' on line {line_number}"
                )
            try:
                value = float(row[1])
            except ValueError as error:
                raise ValueError(f"Invalid rate on line {line_number}") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Rates must be finite and non-negative on line {line_number}"
                )
            if section == "sampling rates" and value > 1:
                raise ValueError(
                    f"Sampling rates must be at most 1 on line {line_number}"
                )
            values[row[0]] = value
        if set(values) != set(states):
            raise ValueError(f"'{section}' must define a value for every state")
        return values

    transition_times = {state: [] for state in states}
    seen_time_states = set()
    for line_number, row in sections["transition times"]:
        if len(row) < 2 or row[0] not in states:
            raise ValueError(f"Invalid transition-times entry on line {line_number}")
        if row[0] in seen_time_states:
            raise ValueError(
                f"Duplicate state in 'transition times' on line {line_number}"
            )
        seen_time_states.add(row[0])
        try:
            transition_times[row[0]] = (
                [] if row[1:] == ["-"] else [float(t) for t in row[1:]]
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid transition time on line {line_number}"
            ) from error
        if any(
            not math.isfinite(t) or t < 0
            for t in transition_times[row[0]]
        ):
            raise ValueError(
                f"Transition times must be finite and non-negative on line "
                f"{line_number}"
            )

    if seen_time_states != set(states):
        raise ValueError("'transition times' must include each state; use '-' for none")

    transition_probabilities = pd.DataFrame(0.0, index=states, columns=states)
    seen_transitions = set()
    for line_number, row in sections["transition probabilities"]:
        if (
            len(row) != 4
            or row[0] not in states
            or row[1] != "->"
            or row[2] not in states
        ):
            raise ValueError(
                f"Invalid transition-probability entry on line {line_number}"
            )
        transition = (row[0], row[2])
        if transition in seen_transitions:
            raise ValueError(f"Duplicate transition on line {line_number}")
        try:
            probability = float(row[3])
        except ValueError as error:
            raise ValueError(
                f"Invalid transition probability on line {line_number}"
            ) from error
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                "Transition probabilities must be finite and between 0 and 1 "
                f"on line {line_number}"
            )
        transition_probabilities.loc[transition[0], transition[1]] = probability
        seen_transitions.add(transition)

    expected_transitions = {(source, target) for source in states for target in states}
    if seen_transitions != expected_transitions:
        raise ValueError(
            "'transition probabilities' must define every state-to-state pair"
        )
    if any(
        not np.isclose(transition_probabilities.loc[state].sum(), 1.0)
        for state in states
    ):
        raise ValueError("Transition probabilities for each state must sum to 1")

    try:
        time_horizon = float(one_value("simulation time horizon"))
        num_trees = int(one_value("num trees"))
    except ValueError as error:
        raise ValueError(
            "Time horizon must be numeric and 'num trees' must be an integer"
        ) from error
    if (
        not math.isfinite(time_horizon)
        or time_horizon <= 0
        or num_trees <= 0
    ):
        raise ValueError("Time horizon and number of trees must be positive")

    root_state = one_value("initial state")
    if root_state not in states:
        raise ValueError("'initial state' must be one of the declared states")

    for state, times in transition_times.items():
        if (
            any(left >= right for left, right in zip(times, times[1:]))
            or any(time > time_horizon for time in times)
        ):
            raise ValueError(
                "Transition times for "
                f"{state} must be strictly increasing and within the time horizon"
            )

    return {
        "states": states,
        "birth_rates": parse_rates("birth rates"),
        "death_rates": parse_rates("death rates"),
        "sampling_rates": parse_rates("sampling rates"),
        "transition_times": transition_times,
        "transition_probabilities": transition_probabilities,
        "time_horizon": time_horizon,
        "root_state": root_state,
        "num_trees": num_trees,
    }


def simulate_from_settings(
    setting_file: str | Path,
    source_seed: int,
    output_dir: str | Path,
    max_attempts: int = 1000,
) -> int:
    """Generate one tree, trying consecutive seeds until it has sampled tips."""
    import matplotlib.pyplot as plt

    from src.likelihood import add_node_times
    from src.plotters import plot_simulated_lineage_tree

    settings = read_simulation_settings(setting_file)
    if source_seed < 0:
        raise ValueError("source_seed must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    differentiation_times = nx.DiGraph()
    for state in settings["states"]:
        differentiation_times.add_node(
            state,
            transition_times=settings["transition_times"][state],
        )
        for target in settings["states"]:
            if (
                state != target
                and settings["transition_probabilities"].loc[state, target] > 0
            ):
                differentiation_times.add_edge(state, target)

    seed = source_seed
    graph = None
    root = None
    for attempt in range(max_attempts):
        seed = source_seed + attempt
        print(f"Trying seed {seed}")
        graph, root = simulate_multitype_bd_with_diff(
                birth_rates=settings["birth_rates"],
                death_rates=settings["death_rates"],
                sampling_rates=settings["sampling_rates"],
                transition_probabilities=settings["transition_probabilities"],
                differentiation_times=differentiation_times,
                time_length=settings["time_horizon"],
                root_state=settings["root_state"],
                seed=seed,
        )
        if any(
                data.get("event") == "sample" for _, data in graph.nodes(data=True)
        ):
                break
        print(f"Tree generation produced no sampled tips with seed {seed}")
    else:
        raise RuntimeError(
                "No sampled extant tips were produced after "
                f"{max_attempts} attempts starting at seed {source_seed}"
        )

    assert graph is not None and root is not None

    graph = add_node_times(graph, root=root, reverse=False)
    for _, data in graph.nodes(data=True):
        data["T"] = max(1 - data["T"] / settings["time_horizon"], 0)

    extant_graph = filter_extant_lineages(graph, root)
    extant_graph, transitions = contract_edges(extant_graph, root)
    export_tree(
        extant_graph,
        output_dir / "nodes.csv",
        output_dir / "edges.csv",
        state_to_id={
            state: index for index, state in enumerate(settings["states"])
        },
    )

    plot_simulated_lineage_tree(graph, root, show_labels=False)
    plt.savefig(output_dir / "T.png")
    plt.close()

    plot_simulated_lineage_tree(extant_graph, root, show_labels=False)
    plt.savefig(output_dir / "T_extant.png")
    plt.close()

    with open(output_dir / "transition_times.csv", "w") as transition_file:
        print(len(transitions), file=transition_file)
        for time, _, _, _, _ in transitions:
            print(time, file=transition_file)

    with open(output_dir / "transition_times_with_edges.csv", "w") as transition_file:
        print(len(transitions), file=transition_file)
        for time, start_node, end_node, source, target in transitions:
            normalized_time = 1 - time / settings["time_horizon"]
            source_id = settings["states"].index(source)
            target_id = settings["states"].index(target)
            print(
                f"{normalized_time},{start_node},{end_node},{source_id},{target_id}",
                file=transition_file,
            )

    return seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate one tree from a settings file."
    )
    parser.add_argument("setting_file", help="Path to the simulator settings file")
    parser.add_argument("source_seed", type=int, help="Seed to try first")
    parser.add_argument(
        "output_dir", help="Directory in which to save the tree outputs"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Maximum consecutive seeds to try when no sampled tips are produced.",
    )
    args = parser.parse_args()

    successful_seed = simulate_from_settings(
        setting_file=args.setting_file,
        source_seed=args.source_seed,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
    )
    print(f"Generated tree using seed {successful_seed}")


if __name__ == "__main__":
    main()
