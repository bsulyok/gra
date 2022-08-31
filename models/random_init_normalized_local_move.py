from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm


def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return dist


def greedy_routing_score(adjacency_matrix: np.ndarray, coord, N, ls, num_grs_iter) -> int:
    vertex_distance = get_distance(coord)
    root = np.empty_like(vertex_distance, dtype=int)
    for vertex in range(N):
        neighbours = np.where(adjacency_matrix[vertex])[0]
        neighbour_dist = vertex_distance[neighbours]
        root[vertex] = neighbours[np.argmin(neighbour_dist, axis=0)]
    np.fill_diagonal(root, ls)
    
    for _ in range(num_grs_iter):
        root = root[root, ls]
    return (np.sum(root==ls)-N)/N/(N-1)


def random_move(coord: np.ndarray, rng, size=1, scale=1) -> np.ndarray:
    N = coord.shape[0]
    x = np.random.choice(coord.shape[0], size=size, replace=False)
    clip_min, clip_max = 1, 2*np.log(N)
    loc = r = coord[x, 0]
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    theta_scale = np.arccos((np.cosh(r)**2-np.cosh(scale))/np.sinh(r)**2)
    coord[x, 1] = np.random.normal(loc=coord[x, 1], scale=theta_scale, size=size) % (2*np.pi)
    coord[x, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
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
    grs = greedy_routing_score(adjacency_matrix, coord, N, ls, num_grs_iter)
    log[0] = grs

    temperature = 1 / np.arange(1, steps+1)
    if verbose:
        temperature = tqdm(temperature)

    # annealing iteration
    for step, temp in enumerate(temperature, start=1):
        
        # compute the new greedy routing score
        new_coord = random_move(coord.copy(), rng, scale=1)        
        new_grs = greedy_routing_score(adjacency_matrix, new_coord, N, ls, num_grs_iter)
        
        # accept or reject the proposed change(s)
        dE = new_grs - grs
        if 0 < dE or np.random.rand() < np.exp(dE / temp):
            coord, grs = new_coord, new_grs
        log[step] = grs
    
    return log, coord