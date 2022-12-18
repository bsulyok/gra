import numpy as np


def initial_distance(coords) -> np.ndarray:
    r, theta = coords.T
    d_theta = np.pi - np.abs(np.pi - np.abs(theta[:, None] - theta))
    distance = np.cosh(r[:, None]) * np.cosh(r) - np.sinh(r[:, None]) * np.sinh(r) * np.cos(d_theta)
    return distance
    d_theta = np.pi - np.abs(np.pi - np.abs(coords[:, 1]-coords[:, np.newaxis, 1]))
    distance = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    distance -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return distance    
    
    
def initial_greedy_closest_neighbour(distance: np.ndarray, adjacency_list: np.ndarray) -> np.ndarray:
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(len(distance)))
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