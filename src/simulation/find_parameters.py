import argparse
import sys
from src.simulator_coord import load_tree
from src.likelihood import add_node_times
from src.mle import find_params
import networkx as nx
import string
import json
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="input tree path", default="T.json")
    parser.add_argument("-o", type=str, help="output path", default="params.json")

    args = parser.parse_args(None if sys.argv[1:] else ["-h"])

    input_tree_path = args.i
    output_path = args.o

    try:

        T, root = load_tree(input_tree_path)

        cell_lineage_T = nx.DiGraph()

        for u, v, data in T.edges(data=True):
            cell_lineage_T.add_edge(u, v, weight=data["length"])

        for n in T.nodes(data=True):
            if n[1]["event"] == "sample":
                cell_lineage_T.nodes[n[0]]["type"] = n[1]["type"]

        cell_lineage_T = add_node_times(cell_lineage_T)

        # diff T with diff time is given
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

        birth_rates, death_rates, transition_probs = find_params(cell_lineage_T, diff_T)

        outputs = {
            "birth_rates": {k: float(v) for k, v in birth_rates.items()},
            "death_rates": {k: float(v) for k, v in death_rates.items()},
            "transition_probabilities": {
                i: {
                    j: float(transition_probs.loc[i, j])
                    for j in transition_probs.columns
                }
                for i in transition_probs.index
            },
        }
    except Exception as e:
        print(e)
        outputs = {}

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)
