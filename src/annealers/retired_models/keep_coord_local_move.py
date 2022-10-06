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
    return root
    #return np.sum(root==ls)-N


def random_move(coord: np.ndarray, rng, size, root, ls) -> np.ndarray:
    N = coord.shape[0]
    x = np.random.choice(coord.shape[0], size=size, replace=False)
    ls = np.arange(N).reshape(1, -1)
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = coord[x, 0], 1
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    coord[x, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
    coord[x, 1] = np.random.normal(loc=coord[x, 1], scale=np.pi/4, size=size) % (2*np.pi)
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

    log = np.empty(steps+1)
    root = greedy_routing_score(adjacency_matrix, coord, N, ls, num_grs_iter)
    grs = np.sum(root==ls)-N
    log[0] = grs

    temperature = 1 / np.arange(1, steps+1)
    if verbose:
        temperature = tqdm(temperature)

    num_sample = 30
        
    # annealing iteration
    for step, temp in enumerate(temperature, start=1):
        
        # compute the new greedy routing score
        new_coord = random_move(coord.copy(), rng, 1, root, ls)
        new_root = greedy_routing_score(adjacency_matrix, new_coord, N, ls, num_grs_iter)
        new_grs = np.sum(new_root==ls)-N
        
        # accept or reject the proposed change(s)
        dE = (new_grs - grs) / N
        if 0 < dE or np.random.rand() < np.exp(dE / temp):
            coord, grs = new_coord, new_grs
            root = new_root
        log[step] = grs
    
    return log/N/(N-1), coord