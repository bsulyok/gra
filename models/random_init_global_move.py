from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm

def greedy_closest_neighbour(adjacency_matrix: np.ndarray, vertex_distance: np.ndarray) -> np.ndarray:
    N = adjacency_matrix.shape[0]
    closest_neighbour = np.empty_like(vertex_distance, dtype=int)
    for vertex in range(N):
        neighbours = np.where(adjacency_matrix[vertex])[0]
        neighbour_dist = vertex_distance[neighbours]
        closest_neighbour[vertex] = neighbours[np.argmin(neighbour_dist, axis=0)]
    np.fill_diagonal(closest_neighbour, np.arange(N))
    return closest_neighbour


def greedy_routing_score(adjacency_matrix: np.ndarray, destination: np.ndarray) -> int:
    N = adjacency_matrix.shape[0]
    ls = np.arange(N)[np.newaxis, :]
    for _ in range(int(np.ceil(np.log2(N-1)))):
        destination = destination[destination, ls]
    return (np.sum(destination==ls)-N)/N/(N-1)


def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return dist


def get_distance_alt(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return dist


def random_move(coord: np.ndarray, rng, size=1) -> np.ndarray:
    x = np.random.choice(coord.shape[0], size=size, replace=False)
    coord[x, 0] = rng()
    coord[x, 1] = 2*np.pi*np.random.rand(size)
    return coord


def greedy_routing_score(adjacency_matrix: np.ndarray, vertex_distance, N, ls, num_grs_iter) -> int:
    root = np.empty_like(vertex_distance, dtype=int)
    for vertex in range(N):
        neighbours = np.where(adjacency_matrix[vertex])[0]
        neighbour_dist = vertex_distance[neighbours]
        root[vertex] = neighbours[np.argmin(neighbour_dist, axis=0)]
    np.fill_diagonal(root, ls)
    
    for _ in range(num_grs_iter):
        root = root[root, ls]
    return (np.sum(root==ls)-N)/N/(N-1)


def load_graph(name: str):
    vertex_data = np.genfromtxt(test_graphs.joinpath(name).with_suffix('.coord'), delimiter=',')
    coord = vertex_data[:, 1:]
    vertex_dict = {vertex: idx for idx, vertex in enumerate(vertex_data[:, 0].astype(int))}
    edge_list = np.genfromtxt(test_graphs.joinpath(name).with_suffix('.edges'), delimiter=',', dtype=int)
    N = coord.shape[0]
    adjacency_matrix = np.zeros((N, N), dtype=bool)
    for source, target in edge_list:
        source, target = vertex_dict[source], vertex_dict[target]
        adjacency_matrix[source, target] = True
        adjacency_matrix[target, source] = True
    return adjacency_matrix, coord


def random_move(coord: np.ndarray, rng, size=1) -> np.ndarray:
    x = np.random.choice(coord.shape[0], size=size, replace=False)
    coord[x, 0] = rng.rvs(size)
    coord[x, 1] = 2*np.pi*np.random.rand(size)
    return coord


def greedy_routing_annealing(
    adjacency_matrix,
    coord,
    steps,
    verbose: bool = False
):

    coord = coord.copy()

    N = len(adjacency_matrix)
    num_grs_iter = int(np.ceil(np.log2(N-1)))
    ls = np.arange(N).reshape(1, -1)
    
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = 2.5*np.log(N), np.log(N) /2
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    rng = truncnorm(a=a, b=b, loc=loc, scale=scale)
    x = rng.rvs(N)
    coord[np.argsort(np.sum(adjacency_matrix, 0)+np.random.rand(N)), 0] = np.sort(x)[::-1]
    coord[:, 1] = 2 * np.pi * np.random.rand(N)

    log = np.empty(steps+1)
    distance = get_distance(coord)
    destination = greedy_closest_neighbour(adjacency_matrix, distance)
    grs = greedy_routing_score(adjacency_matrix, distance, N, ls, num_grs_iter)
    log[0] = grs

    iter_rounds = range(1, steps+1)
    if verbose:
        iter_rounds = tqdm(iter_rounds)

    # annealing iteration
    for step in iter_rounds:
        
        # compute the new greedy routing score
        new_coord = random_move(coord.copy(), rng, 1)
        distance = get_distance(new_coord)
        
        new_grs = greedy_routing_score(adjacency_matrix, distance, N, ls, num_grs_iter)
        
        # accept or reject the proposed change(s)
        dE = new_grs - grs
        temp = 1 / step
        if 0 < dE or np.random.rand() < np.exp(dE / temp):
            coord, grs = new_coord, new_grs
        log[step] = grs
    
    return log, coord