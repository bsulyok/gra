from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm

def get_random_init(adjacency_matrix):
    N = adjacency_matrix.shape[0]
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = 2.5*np.log(N), np.log(N) /2
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    rng = truncnorm(a=a, b=b, loc=loc, scale=scale)
    new_r = rng.rvs(N)
    new_r = np.sort(new_r)[::-1]
    new_theta = 2 * np.pi * np.random.rand(N)
    degree_order = np.argsort(np.sum(adjacency_matrix, 0)+np.random.rand(N))
    coord = np.empty((N, 2))
    coord[degree_order, 0] = new_r
    coord[:, 1] = new_theta
    return coord

def select_for_move_random(gd: np.ndarray) -> np.ndarray:
    N = gd.shape[0]
    p = N + 1 - np.sum(gd.T==np.arange(N), axis=1)
    vertex = np.random.choice(N, replace=False, p=p / p.sum())
    return vertex

def select_for_move_target_dependet(gd: np.ndarray) -> np.ndarray:
    N = gd.shape[0]
    p = N + 1 - np.sum(gd.T==np.arange(N), axis=1)
    return np.random.choice(N, replace=False, p=p / p.sum())

def select_for_move_clogging_dependent(gd: np.ndarray) -> np.ndarray:
    N = gd.shape[0]
    unique, counts = np.unique(gd[gd != np.arange(N)], return_counts=True)
    return np.random.choice(unique, p=counts/counts.sum())

def random_move(coord: np.ndarray, moved_vertex: int) -> np.ndarray:
    N = coord.shape[0]
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = coord[moved_vertex, 0], 1
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    coord[moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
    coord[moved_vertex, 1] = np.random.normal(loc=coord[moved_vertex, 1], scale=np.pi/4) % (2*np.pi)
    return coord

def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return dist

def recalc_distance(coord: np.ndarray, distance: np.ndarray, moved_vertex: int) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[moved_vertex, 1]-coord[:, 1]))
    dist = np.cosh(coord[moved_vertex, 0]) * np.cosh(coord[:, 0])
    dist -= np.sinh(coord[moved_vertex, 0]) * np.sinh(coord[:, 0]) * np.cos(d_theta)
    dist[moved_vertex] = 1
    distance[moved_vertex] = distance[:, moved_vertex] = dist
    return distance

def greedy_closest_neighbour(adjacency_list: np.ndarray, distance: np.ndarray) -> np.ndarray:
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(distance.shape[0]))
    return gcn

def greedy_destination(gcn: np.ndarray, num_grs_iter: int):
    arange = np.arange(gcn.shape[1])
    for _ in range(num_grs_iter):
        gcn = gcn[gcn, arange]
    return gcn

def recalc_greedy_routing(adjacency_list, distance, num_grs_iter, gcn, gd, moved_vertex):
    N = adjacency_list.shape[0]
    
    affected = np.append(adjacency_list[moved_vertex], moved_vertex)
    affected_next_hop = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list[affected]])
    affected_next_hop[np.arange(affected.shape[0]), affected] = affected
    changed = np.any(affected_next_hop != gcn[affected], axis=0)
    new_gcn = gcn.copy()
    
    new_gcn[:, moved_vertex] = np.array([neighbours[np.argmin(distance[neighbours, moved_vertex], axis=0)] for neighbours in adjacency_list])
    new_gcn[affected] = affected_next_hop
    
    changed[moved_vertex] = True
    new_gd = gd.copy()
    new_gd[:, changed] = greedy_destination(new_gcn[:, changed], num_grs_iter)
    return new_gcn, new_gd

def greedy_routing_score(gd: np.ndarray) -> float:
    return np.sum(gd==np.arange(gd.shape[0])) - gd.shape[0]

def embed(adjacency_matrix, coord=None, steps=2**8, verbose=False):
    
    adjacency_list = np.array([np.where(neigh)[0] for neigh in adjacency_matrix], dtype=object)
    N = adjacency_matrix.shape[0]
    num_grs_iter = int(np.ceil(np.log2(N-1)))
    
    if coord is None:
        coord = get_random_init(adjacency_matrix)

    steps = int(steps)
    log = np.empty(steps+1)

    dist = get_distance(coord)
    gcn = greedy_closest_neighbour(adjacency_list, dist)
    gd = greedy_destination(gcn, num_grs_iter)
    grs = greedy_routing_score(gd)
    log[0] = grs

    temperature = 1 / np.arange(1, steps+1)
    if verbose:
        temperature = tqdm(temperature)

    for step, temp in enumerate(temperature, start=1):
        moved_vertex = select_for_move_target_dependet(gd)
        new_coord = random_move(coord.copy(), moved_vertex)
        new_dist = recalc_distance(new_coord.copy(), dist.copy(), moved_vertex)
        new_gcn, new_gd = recalc_greedy_routing(adjacency_list, new_dist, num_grs_iter, gcn.copy(), gd.copy(), moved_vertex)
        new_grs = greedy_routing_score(new_gd)

        dE = (new_grs - grs) / N

        if 0 < dE or np.random.rand() < np.exp(dE / temp):
            coord, grs, gd, dist, gcn = new_coord, new_grs, new_gd, new_dist, new_gcn
        log[step] = grs

    return log/N/(N-1), coord