import pandas as pd
import numpy as np
import networkx as nx
import re
from newick import read
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import argparse


from src.likelihood import add_node_times, assign_states_to_internal_nodes


DATA_ROOT_DIR = "data/icefase/data"

import json

def get_fatemap(fatemap_id):

    with open(f"{DATA_ROOT_DIR}/fatemaps/fate_map{fatemap_id:04d}.json") as f:
        diff_T = nx.DiGraph()

        fatemap = json.load(f)

        all_nodes = fatemap["all_id"]
        for node in all_nodes:
            diff_T.add_node(node)
        
        merges = fatemap["merge"]
        for root, children in merges.items():
            for child in children:
                diff_T.add_edge(root, child)
        
        diff_times = fatemap["diff_time"]
        for node, time in diff_times.items():
            if time == "Inf": 
                diff_T.nodes[node]["diff_time"] = None
            else:
                diff_T.nodes[node]["diff_time"] = time
        
        diff_T.graph['root'] = fatemap["root_id"]
        
        transition_probs = pd.DataFrame(np.zeros((len(all_nodes), len(all_nodes))), index=all_nodes, columns=all_nodes)
        diff_mode_probs = fatemap["diff_mode_probs"]

        for root, children in merges.items():
            qs = diff_mode_probs[root]
            for children, q in zip(children, qs):
                transition_probs.loc[root, children] = q

    return diff_T, transition_probs


def tree_to_nx(tree):
    G = nx.DiGraph()

    def node_name(n):
        return n.name if n.name else str(id(n))
    
    # parse cell_type from node name
    def parse_cell_type(name):
        # assumes format: type_<cell_type>_gen_...
        m = re.match(r"type_(-?\d+)_gen_(-?\d+)_(-?\d+)", name)
        if m:
            return str(m.group(1))
        else:
            return None

    root_name = node_name(tree)
    G.graph["root"] = root_name   # store root

    def add_node(node):
        parent = node_name(node)
        G.add_node(parent, state=parse_cell_type(parent))

        for child in node.descendants:
            child_name = node_name(child)

            G.add_edge(
                parent,
                child_name,
                weight=child.length,
            )

            add_node(child)

    add_node(tree)
    return G


def get_predicted_diff_times(diff_T, T):
    diff_times = {}

    for node in nx.dfs_postorder_nodes(diff_T, source=diff_T.graph['root']):
        if node[0] != "-": # internal node
            diff_time = 0

            children_states = list(diff_T.successors(node))

            nodes_with_children_state = [n for n, d in T.nodes(data=True) if d.get("state") in children_states]
            their_parent_nodes = list(set([list(T.predecessors(n))[0] for n in nodes_with_children_state]))
            
            for pnode in their_parent_nodes:
                children_nodes = list(T.successors(pnode))
                children_node_states = [T.nodes[child].get("state") for child in children_nodes]

                if len(set(children_node_states) & set(children_states)) > 1:
                    diff_time = max(diff_time, T.nodes[pnode]['T'])

            diff_times[node] = diff_time
    
    EPS = 1e-9

    for node in nx.dfs_preorder_nodes(diff_T, source=diff_T.graph['root']):
        if node[0] != "-":
            children_states = list(diff_T.successors(node))
            for child_state in children_states:
                if child_state[0] != "-":
                    diff_times[child_state] = max(diff_times[child_state], diff_times[node]) + EPS

    corrected_diff_T = diff_T.copy()

    for state, diff_time in diff_times.items():
        corrected_diff_T.nodes[state]['diff_time'] = diff_time + EPS
        # corrected_diff_T.nodes[state]['diff_time'] = (diff_time + EPS) / 11.5
    
    return corrected_diff_T



def pull_down_diff_times(diff_T, stretch=0.2):
    # expects diff times to be within (0, 1)
    diff_T = diff_T.copy()
    for n in nx.dfs_postorder_nodes(diff_T, source=diff_T.graph['root']):
        if not n.startswith("-"):
            children = list(diff_T.successors(n))
            children_diff_times = []
            for c in children:
                if c.startswith("-"): children_diff_times.append(1)
                else: children_diff_times.append(diff_T.nodes[c]['diff_time'])
            
            node_diff_time = diff_T.nodes[n]['diff_time']
            min_children_diff_time = min(children_diff_times)
            pulled_down_diff_time = node_diff_time + (min_children_diff_time - node_diff_time) * stretch
            print(n, children, children_diff_times, min_children_diff_time, node_diff_time, pulled_down_diff_time)
            diff_T.nodes[n]['diff_time'] = pulled_down_diff_time
    
    return diff_T


def predict_commitment_bias(id):
    metadata = pd.read_csv(f"{DATA_ROOT_DIR}/experiment_metadata.csv")

    eid = metadata.loc[id, "Experiment id"]
    fatemap_id = metadata.loc[id, "Fate map id"]

    print("id:\t\t", id)
    print("experiment id:\t", eid)
    print("fatemap id:\t", fatemap_id)

    diff_T, transition_probs = get_fatemap(fatemap_id)

    tree = read(f"{DATA_ROOT_DIR}/simulated_phylogeny/{eid:04d}.newick")[0]

    T = tree_to_nx(tree)
    T = add_node_times(T, T.graph['root'], reverse=False)

    predicted_diff_T = get_predicted_diff_times(diff_T, T)
    predicted_diff_T = pull_down_diff_times(predicted_diff_T, stretch=1/2)


    ## flip times
    rev_T = T.copy()
    for n in T.nodes():
        rev_T.nodes[n]['T'] = (T.nodes[n]['T']) / 11.5

    for u, v, data in rev_T.edges(data=True):
        data['weight'] /= 11.5    

    rev_T2 = rev_T.copy()

    for n in rev_T.nodes():
        rev_T2.nodes[n]['T'] = (1 - rev_T.nodes[n]['T']) 
    
    rev_diff_T = predicted_diff_T.copy()
    for n in predicted_diff_T.nodes():
        diff_time = rev_diff_T.nodes[n].get('diff_time')
        if diff_time is None: rev_diff_T.nodes[n]['diff_time'] = 0
        else: rev_diff_T.nodes[n]['diff_time'] = 1 - (rev_diff_T.nodes[n]['diff_time'] / 11.5)

    rev_T2 = assign_states_to_internal_nodes(rev_T2, rev_diff_T, correct=True)

    types = list(diff_T.nodes())

    transition_maps = [
        (i, succ)
        for i in diff_T.nodes()
        if (succ := list(diff_T.successors(i)))
    ]

    def find_params_simple(trees, diff_T):

        state_to_idx = {state: i for i, state in enumerate(types)}
        
        q = np.zeros((len(types), len(types)))

        for cell_lineage_T in trees:

            for u, v in cell_lineage_T.edges():
                u_state = cell_lineage_T.nodes[u]['state']
                v_state = cell_lineage_T.nodes[v]['state']

                path = nx.shortest_path(diff_T, u_state, v_state)

                for i in range(len(path) - 1):
                    u_idx = state_to_idx[path[i]]
                    v_idx = state_to_idx[path[i+1]]
                    q[u_idx, v_idx] += 1

        # Normalize row-wise (handle zero rows safely)
        row_sums = q.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero

        q_normalized = q / row_sums

        Q = pd.DataFrame(q_normalized, index=types, columns=types)
        
        return Q

    
    trees = [rev_T2]
    Q = find_params_simple(trees, rev_diff_T)
    Q.to_csv(f"outputs/icefase/commitment_bias_predictions_50/{eid}/q.csv")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)

    args = parser.parse_args()

    i = args.id - 1
    predict_commitment_bias(i)
