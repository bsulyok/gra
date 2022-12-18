import numpy as np
import networkx as nx
from scipy import sparse as sp
from itertools import combinations, pairwise
from scipy import stats


def get_distance(r_u, r_v, theta_u, theta_v):
    d_theta = np.pi-np.abs(np.pi-np.abs(theta_u-theta_v))
    return np.arccosh(np.cosh(r_u)*np.cosh(r_v)-np.sinh(r_u)*np.sinh(r_v)*np.cos(d_theta))


def initial_greedy_closest_neighbour(distance: np.ndarray, adjacency_list: np.ndarray) -> np.ndarray:
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(len(distance.shape)))
    return gcn


def initial_greedy_destination(gcn: np.ndarray) -> np.ndarray:
    N = len(gcn)
    iter_count = int(np.ceil(np.log2(N-1)))
    gd = gcn.copy()
    arange = np.arange(len(gd))
    for _ in range(iter_count):
        gd = gd[gd, arange]
    return gd
    
    
def greedy_routing_score(gd: np.ndarray) -> float:
    N = len(gd)
    successful = np.sum(gd == np.arange(N))
    return (successful - N) / N / (N-1)


def greedy_routing_score(G, coords):
    pass


def geometrical_congruence(G, coords):
    r, theta = coords.T
    spadjm = nx.adjacency_matrix(G)
    N = spadjm.shape[0]
    u, v = spadjm.nonzero()
    spdistm = sp.csr_matrix((get_distance(r[u], r[v], theta[u], theta[v]), (u, v)), shape=spadjm.shape)
    r, theta = coords.T
    score = 0
    num_nonadjacent = N*(N-1) - spadjm.nnz / 2
    for source, target in combinations(range(N), 2):
        if spadjm[source, target] == 1:
            continue
        ptsp = np.mean([sum(spdistm[i, j] for i, j in pairwise(path)) for path in nx.all_shortest_paths(G, source, target)])
        dist = get_distance(r[source], r[target], theta[source], theta[target])
        score += dist / ptsp
    return score / num_nonadjacent


def mapping_accuracy(G, coords):
    r, theta = coords.T
    spadjm = nx.adjacency_matrix(G)
    N = spadjm.shape[0]
    u, v = spadjm.nonzero()
    spdistm = sp.csr_matrix((get_distance(r[u], r[v], theta[u], theta[v]), (u, v)), shape=spadjm.shape)
    r, theta = coords.T
    dist, spl = [], []
    shortest_path_length = dict(nx.shortest_path_length(G))
    for source, target in combinations(range(N), 2):
        spl.append(shortest_path_length[source][target])
        dist.append(get_distance(r[source], r[target], theta[source], theta[target]))
    return stats.spearmanr(spl, dist).correlation