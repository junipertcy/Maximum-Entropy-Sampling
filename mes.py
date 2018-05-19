"""
## Maximum Entropy Sampling Algorithm
The code below implements the maximum entropy sampling algorith. It makes use of the submodularity of the entropy
function to implement a lazy greedy search. A rooted tree in a potential node to sample is constructed using a
Dijkstra-like algorithm. The main function returns two dictionaries with keys given by the labels of the nodes.

One provides information on the rank of the nodes as they are added to the set of sampled nodes; the other dictionary
provides information on the value of the joint entropy of the set of observed nodes when the node was included in the
set of observed nodes. The main function returns also a list containing the ordered list of nodes that composed the set
of observed nodes. The algorithm takes as inputs a graph and the dictionaries of the marginal and conditional
pairwise entropy.

We provide also functions that construct the set of observed nodes using marginal entropy only, and randomly.
"""

import heapq
import random
import time


def compute_score_dijkstra(i, G, cond_entropy, observed, vector_observed):
    """
    Compute the sum of pairwise conditional entropies (Eq. 8 in the PRL paper), given a set of observed nodes.
    Note that each observed node contributes one term in the sum.

    Parameters
    ----------
    i: `int`
        The root node (source node) to perform the dijkstra algorithm for all the other nodes (target nodes).
    G: `networkx graph object`
        The underlying sparse graph
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.
    observed: `dict`
        The dict of observed nodes.
    vector_observed: `dict`
        The dictionary to indicate whether a node has been observed or not.

    Returns
    -------
    score: `float`
        Sum of pairwise conditional entropies.

    """
    N = float(len(G))

    source = {}
    distance = {}

    for n in G:
        distance[n] = N + 10
        source[n] = None

    distance[i] = 0.0
    source[i] = i

    # heap##########
    heap_distance = []
    for n in G:
        tmp = distance[n]
        heapq.heappush(heap_distance, (tmp, n))
    ##################

    unvisited = {}
    vector_unvisited = {}
    for n in G:
        unvisited[n] = 1
        vector_unvisited[n] = -1

    while len(unvisited) > 0:

        control = -1
        while control < 0:
            dist_current, current = heapq.heappop(heap_distance)
            if vector_unvisited[current] < 0:
                control = 1

        vector_unvisited[current] = 1
        unvisited.pop(current, None)

        neigh = G.neighbors(current)
        # random.shuffle(neigh)

        for m in neigh:

            if vector_unvisited[m] < 0:

                if vector_observed[m] > 0:
                    if source[m] is None:
                        source[m] = source[current]
                        if vector_observed[current] > 0 or current == i:
                            source[m] = current
                        distance[m] = distance[current] + cond_entropy[m, source[m]]

                    else:
                        if vector_observed[current] > 0 or current == i:
                            # delta = cond_entropy[m, current] - cond_entropy[m, source[m]]
                            delta = distance[current] + cond_entropy[m, current] - distance[m]
                            if delta < 0:
                                source[m] = current
                                distance[m] = distance[current] + cond_entropy[m, current]
                        else:
                            # delta = cond_entropy[m, source[current]] - cond_entropy[m, source[m]]
                            delta = distance[current] + cond_entropy[m, source[current]] - distance[m]
                            if delta < 0:
                                source[m] = source[current]
                                distance[m] = distance[current] + cond_entropy[m, source[m]]

                else:
                    source[m] = source[current]
                    if vector_observed[current] > 0 or current == i:
                        source[m] = current
                    distance[m] = distance[current]

                heapq.heappush(heap_distance, (distance[m], m))

    score = 0.0
    for m in observed:
        score = score + cond_entropy[m, source[m]]

    return score


def lazy_find_dijkstra_tree(G, observed, vector_observed, heap_entropy, value_entropy, marg_entropy, cond_entropy,
                            old_entropy):
    """
    Find the node and its associated conditional entropy (given observed nodes) when one knows the graph structure,
    marginal entropies, and the conditional pairwise entropies.
    The sparse network is reduced to a tree by the `compute_score_dijkstra` method.

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    observed: `dict`
        The dict of observed nodes.
    vector_observed: `dict`
        The dictionary to indicate whether a node has been observed or not.
    heap_entropy: `list`
        The marginal entropies for all nodes, represented as a heap.
    value_entropy: `dict`
        The marginal entropies for all nodes.
    marg_entropy: `dict`
        The marginal entropy of each node
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.
    old_entropy: `float`
        The computed pairwise conditional entropy at the previous stage, i.e. the last term in Eq. 7 of the PRL paper.

    Returns
    -------
    node_max: `int`
        The node to query, following the list of observed nodes.
    _max_value: `float`
        The conditional entropy, H(node_max|observed); i.e. the left hand side of Eq. 7 of the PRL paper.
    """
    # lazy search
    max_value, node_max = heap_entropy[0]

    new_node_max = -1
    while new_node_max != node_max:
        node_max = new_node_max

        new_max_value, new_node_max = heapq.heappop(heap_entropy)

        tmp_value = compute_score_dijkstra(new_node_max, G, cond_entropy, observed, vector_observed)

        value_entropy[new_node_max] = tmp_value + marg_entropy[new_node_max] - old_entropy
        # break ties
        value_entropy[new_node_max] += 1e-20 * (0.5 - random.random())

        tmp = (- value_entropy[new_node_max], new_node_max)
        heapq.heappush(heap_entropy, tmp)

    # removal node
    tmp = heapq.heappop(heap_entropy)
    node_max = tmp[1]
    max_value = tmp[0]

    value_entropy.pop(node_max, None)
    _max_value = - max_value

    return node_max, _max_value


def max_ent_sampling(G, marg_entropy, cond_entropy):
    """
    Obtain a ranked list of nodes, acquired by maximum entropy sampling.

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    marg_entropy: `dict`
        The marginal entropy of each node
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.

    Returns
    -------
    rank: `dict`
        The rank of each node that should be queried, according to MES.
    value: `dict`
        The condition entropy of a node, conditioned on a full knowledge of the set of observed nodes.
    list_observed: `list`
        The list of nodes that are observed orderly, according to MES. This is similar to `rank`.

    """
    start_time = time.time()

    # for lazy search
    heap_entropy = []
    value_entropy = {}
    for n in G:
        value_entropy[n] = marg_entropy[n]
        tmp = (- value_entropy[n], n)
        heapq.heappush(heap_entropy, tmp)

    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
    rank = {}
    value = {}
    r = 1

    while total_observed < N:
        node, ent = lazy_find_dijkstra_tree(G, observed, vector_observed, heap_entropy, value_entropy, marg_entropy,
                                            cond_entropy, old_entropy)
        old_entropy = old_entropy + ent
        # print (node, ent, old_entropy)

        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1

    print("--- %s seconds ---" % (time.time() - start_time))

    return rank, value, list_observed


def max_ent_sampling_ind_approx(G, marg_entropy, cond_entropy):
    """
    Obtain a ranked list of nodes, acquired by the individual-based mean-field (IBMF) approximation.

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    marg_entropy: `dict`
        The marginal entropy of each node
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.

    Returns
    -------
    rank: `dict`
        The rank of each node that should be queried, according to MES.
    value: `dict`
        The condition entropy of a node, conditioned on a full knowledge of the set of observed nodes.
    list_observed: `list`
        The list of nodes that are observed orderly, according to MES. This is similar to `rank`.

    """
    start_time = time.time()

    # for lazy search
    heap_entropy = []
    value_entropy = {}
    for n in G:
        value_entropy[n] = marg_entropy[n]
        tmp = (- value_entropy[n], n)
        heapq.heappush(heap_entropy, tmp)

    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
    rank = {}
    value = {}
    r = 1

    while total_observed < N:
        tnew_max_valuemp, node = heapq.heappop(heap_entropy)

        tmp_value = compute_score_dijkstra(node, G, cond_entropy, observed, vector_observed)
        ent = tmp_value + marg_entropy[node] - old_entropy

        old_entropy = old_entropy + ent
        # print (node, ent, old_entropy)

        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1

    print("--- %s seconds ---" % (time.time() - start_time))

    return rank, value, list_observed


def random_sampling(G, marg_entropy, cond_entropy):
    """
    Obtain a ranked list of nodes, acquired randomly.

    Parameters
    ----------
    G: `networkx graph object`
        The underlying sparse graph
    marg_entropy: `dict`
        The marginal entropy of each node
    cond_entropy: `dict`
        The conditional pairwise entropy, `H(i|j)`, of node `i` conditioned on knowing node `j`.
        It is accessed via `cond_entropy[i, j]`, with `i` and `j` being node indexes.

    Returns
    -------
    rank: `dict`
        The rank of each node that should be queried, according to MES.
    value: `dict`
        The condition entropy of a node, conditioned on a full knowledge of the set of observed nodes.
    list_observed: `list`
        The list of nodes that are observed orderly, according to MES. This is similar to `rank`.

    """
    start_time = time.time()

    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    unobserved = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
        unobserved[n] = 1
    rank = {}
    value = {}
    r = 1

    while total_observed < N:
        node = random.choice(list(unobserved.keys()))

        tmp_value = compute_score_dijkstra(node, G, cond_entropy, observed, vector_observed)
        ent = tmp_value + marg_entropy[node] - old_entropy

        unobserved.pop(node, None)

        old_entropy = old_entropy + ent
        # print (node, ent, old_entropy)

        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1

    print("--- %s seconds ---" % (time.time() - start_time))

    return rank, value, list_observed
