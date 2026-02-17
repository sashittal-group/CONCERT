import numpy as np
import pandas as pd
from src.likelihood import calculate_likelihood_of_tree
from scipy.optimize import minimize


def unpack_params(theta):
    types = ['A', 'B', 'C']

    k = 0

    death = {}
    for t in types:
        death[t] = np.exp(theta[k])
        k += 1

    # Birth rates (strictly greater than death)
    birth = {}
    for t in types:
        birth[t] = death[t] + np.exp(theta[k])
        k += 1

    # Transition matrix via softmax
    Q = pd.DataFrame(0.0, index=types, columns=types)

    transition_maps = [
        ('A', ['B', 'C']),
    ]

    for i, transition_to in transition_maps:
        logits = []
        js = []

        for j in transition_to:
            logits.append(theta[k])
            js.append(j)
            k += 1

        logits = np.array(logits)
        logits -= np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        for idx, j in enumerate(js):
            Q.loc[i, j] = probs[idx]

    return birth, death, Q

def neg_log_likelihood(theta, cell_lineage_T, diff_T, sampling_rate=1.0):
    birth, death, Q = unpack_params(theta)

    p = calculate_likelihood_of_tree(
        birth_rates=birth,
        death_rates=death,
        sampling_rate=sampling_rate,
        transition_probabilities=Q,
        cell_lineage_T=cell_lineage_T,
        diff_T=diff_T
    )

    # if p <= 1e-300:
    #     return 1e6 

    return -np.log(p)

def pack_initial():
    params = []
    types = ['A', 'B', 'C']
    transition_maps = [
        ('A', ['B', 'C']),
    ]

    # log death
    for t in types:
        params.append(np.log(0.5))

    # log birth
    for t in types:
        params.append(np.log(2))

    # transition logits (inverse softmax approx)
    for i, transition_to in transition_maps:
        for j in transition_to:
            params.append(np.log(0.5))

    return np.array(params)


def find_params(cell_lineage_T, diff_T):
    theta0 = pack_initial()

    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(cell_lineage_T, diff_T, 1.0),
        method="L-BFGS-B"
    )

    return unpack_params(result.x)
