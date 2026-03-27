import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import os
import json
import argparse
from src.plotters import plot_differentiation_tree, plot_simulated_lineage_tree
from src.simulator import simulate_multitype_bd_with_diff, save_tree, filter_extant_lineages, load_tree
from src.likelihood import add_node_times, assign_states_to_internal_nodes

from math import exp, log
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

MAX_BIRTH = 50.0


def calculate_log_likelihood(T, birth_rate, death_rate, sampling_rate=1):
    ll = 0.0
    
    ll += len(T.nodes()) * log(birth_rate)

    diff_time = 0
    U = 1 - sampling_rate
    C = (U - 1) / (U - (death_rate / birth_rate))

    for u, v in T.edges():
        u_t = T.nodes[u]['T']
        v_t = T.nodes[v]['T']
        assert u_t > v_t

        flow = exp( -(birth_rate - death_rate) * (u_t - v_t)) * \
                    ( (C - exp( -(birth_rate - death_rate) * (v_t - diff_time))) / 
                      (C - exp( -(birth_rate - death_rate) * (u_t - diff_time))) ) ** 2
        
        assert not np.isnan(flow)

        ll += log(flow)

    return ll


def unpack_params(theta):

    k = 0

    birth = MAX_BIRTH * sigmoid(theta[k])
    k += 1

    return birth


def neg_log_likelihood(theta, cell_lineage_T, sampling_rate=1.0):

    birth = unpack_params(theta)
    
    l = calculate_log_likelihood(
        T = cell_lineage_T,
        birth_rate=birth,
        death_rate=1e-30,
        sampling_rate=sampling_rate,
    )

    return -l


def pack_initial():

    params = []
    params.append(0.0)  # sigmoid(0)=0.5 → birth ≈ halfway to MAX_BIRTH

    return np.array(params)


def find_params(cell_lineage_T):

    theta0 = pack_initial()

    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(cell_lineage_T, 1.0),
        method="L-BFGS-B"
    )

    birth = unpack_params(result.x)

    return birth


def predict_birth(path):
    T, root = load_tree(f"{path}/T_extant.json")

    cell_lineage_T = nx.DiGraph()

    for u, v, data in T.edges(data=True):
        cell_lineage_T.add_edge(u, v, weight=data['length'])

    for n, d in T.nodes(data=True):
        if d['event'] == 'sample':
            cell_lineage_T.nodes[n]['state'] = d['type']
    
    cell_lineage_T = add_node_times(cell_lineage_T, reverse=False)
    T = cell_lineage_T
    rev_T = T.copy()
    for n in T.nodes():
        rev_T.nodes[n]['T'] = (T.nodes[n]['T']) / 5

    for u, v, data in rev_T.edges(data=True):
        data['weight'] /= 5

    rev_T2 = rev_T.copy()

    for n in rev_T.nodes():
        rev_T2.nodes[n]['T'] = (1 - rev_T.nodes[n]['T'])
    
    diff_T = nx.DiGraph()

    diff_T.add_node("A", diff_time=None)

    rev_T2 = assign_states_to_internal_nodes(rev_T2, diff_T, correct=True)

    birth_rate = find_params(rev_T2)

    dict = {
        "birth_rate": birth_rate
    }

    os.makedirs(f"{path}/birth", exist_ok=True)
    with open(f"{path}/birth/inferred_parameters.json", "w") as f:
        json.dump(dict, fp=f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    path = args.path

    predict_birth(path)