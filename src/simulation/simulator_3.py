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


def simulate(input_dir, seed):

    root_dir = input_dir
    diff_T, _ = load_tree(f"{root_dir}/diff_T.json")

    Qs = pd.read_csv(f"{root_dir}/Q.csv", index_col=0)
    print(Qs)

    save_dir = f"{root_dir}/{seed}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{root_dir}/params.json") as f:
        params = json.load(f)
        lam = params["birth_rates"]
        mu = params["death_rates"]


    G, root = simulate_multitype_bd_with_diff(
        birth_rates=lam,
        death_rates=mu,
        differentiation_times=diff_T,
        transition_probabilities=Qs,
        time_length=5,
        root_type="A",
        seed=seed,
    )

    # for u, v, data in G.edges(data=True):
    #     print((u, v), data)

    save_tree(G, root, f"{save_dir}/T.json")
    plot_simulated_lineage_tree(G, root)
    plt.savefig(f"{save_dir}/T.png")
    plt.close()

    G_extant = filter_extant_lineages(G, root)

    save_tree(G_extant, root, f"{save_dir}/T_extant.json")

    plot_simulated_lineage_tree(G_extant, root)
    plt.savefig(f"{save_dir}/T_extant.png")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    seed = args.seed

    simulate(input_dir, seed)
