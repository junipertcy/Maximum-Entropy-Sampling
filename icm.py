"""
## Independent Cascade (IC) model

Marginal and pairwise entropy of variables are computed using $T$ independent numerical simulations of the IC model.
The model has a single parameter $p$ standing for the probability of spreading. Each numerical simulations is started
from a configuration where all nodes are in the susceptible state, except for a single randomly chosen seed in the
infected state. We compute the probability of the microscopic configuration obtained at the end of the dynamics, when
there are no longer infected nodes.

"""

import random
import numpy as np


def single_simulation_ICM(G, p):
    """
    Perform a single simulation of Independent Cascade on a network.

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    p: `float`
        The probability of spreading

    Returns
    -------
    state: `dict`
        The status of which nodes are infected and which are not

    """

    state = {}
    active = []
    for n in G:
        state[n] = 0  # its node is not infected

    # one random seed active
    n = random.choice(list(G.nodes()))
    state[n] = 1  # its node is infectious
    active.append(n)

    while len(active) > 0:
        tmp = []
        for i in range(0, len(active)):
            n = active[i]
            neigh = G.neighbors(n)
            for m in neigh:
                if state[m] == 0:
                    if random.random() < p:  # otherwise, the node remains uninfected
                        state[m] = 1
                        tmp.append(m)
            state[n] = 2  # its node is infected (and transmitted to its neighbors)

        active = []
        active[:] = tmp[:]

    return state


def multiple_simulations_ICM(G, p, T):
    """
    Perform several independent ICM runs to collect the marginal and conditional pairwise entropies of the stochasitic
    process (i.e. independent cascade).

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    p: `float`
        The probability of spreading
    T: `int`
        Number of independent `single_simulation_ICM` runs.

    Returns
    -------
    marg_entropy: `dict`
        The marginal entropy of each node
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.


    """
    list_of_nodes = sorted(list(G.nodes()))

    joint_prob = {}
    marg_prob = {}
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        marg_prob[n] = {}
    for i in range(0, len(list_of_nodes) - 1):
        n = list_of_nodes[i]
        for j in range(i + 1, len(list_of_nodes)):
            m = list_of_nodes[j]
            joint_prob[n, m] = {}

    results = {}
    for t in range(0, T):
        results[t] = single_simulation_ICM(G, p)

    for t in range(0, T):
        for i in range(0, len(list_of_nodes)):
            n = list_of_nodes[i]
            res = results[t][n]
            if res not in marg_prob[n]:
                marg_prob[n][res] = 0.0
            marg_prob[n][res] += 1.0 / float(T)

        for i in range(0, len(list_of_nodes) - 1):
            n = list_of_nodes[i]
            for j in range(i + 1, len(list_of_nodes)):
                m = list_of_nodes[j]
                res = str(results[t][n]) + ',' + str(results[t][m])
                if res not in joint_prob[n, m]:
                    joint_prob[n, m][res] = 0.0
                joint_prob[n, m][res] += 1.0 / float(T)

    marg_entropy = {}
    cond_entropy = {}

    for n in marg_prob:
        marg_entropy[n] = 0.0
        for z in marg_prob[n]:
            tmp = marg_prob[n][z]
            if tmp > 0.0:
                marg_entropy[n] -= tmp * np.log2(tmp)

    for n, m in joint_prob:
        cond_entropy[n, m] = 0.0
        cond_entropy[m, n] = 0.0
        for z in joint_prob[n, m]:
            tmp = joint_prob[n, m][z]
            if tmp > 0.0:
                cond_entropy[n, m] -= tmp * np.log2(tmp)
                cond_entropy[m, n] -= tmp * np.log2(tmp)
        cond_entropy[n, m] -= marg_entropy[m]
        cond_entropy[m, n] -= marg_entropy[n]

    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        cond_entropy[n, n] = 0.0

    return marg_entropy, cond_entropy
