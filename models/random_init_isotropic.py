from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm, normal


def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 2]-coord[:, np.newaxis, 2]))
    dist = coord[:, 0] * coord[:, np.newaxis, 0]
    dist -= coord[:, 1] * coord[:, np.newaxis, 1] * np.cos(d_theta)
    np.fill_diagonal(dist, 1)
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


def random_move(
    coord: np.ndarray,
    scale=1000,
    t_max=None,
    r_max=None,
    adjacency_matrix=None
) -> np.ndarray:
    
    N = coord.shape[0]
    v = np.random.randint(N)
    t_v, r_v, theta_v = coord[v]
    x, y = r_v * np.cos(theta_v), r_v * np.sin(theta_v)
    M = np.array([
        [t_v, x, y],
        [x, x*x/(t_v+1)+1, x*y/(t_v+1)],
        [x, x*y/(t_v+1), y*y/(t_v+1)+1]
    ])
    
    #distance = get_distance(coord)
    #neigh_dist = distance[v][adjacency_matrix[v]]
    #neigh_dist = np.arccosh(neigh_dist)
    #scale = neigh_dist.max()
    dist_move = np.random.normal(scale=scale)
    theta_move = 2 * np.pi * np.random.rand()
    t_move, r_move = np.cosh(dist_move), np.sinh(dist_move)
    move = np.array([t_move, r_move*np.cos(theta_move), r_move*np.sin(theta_move)])
    t, x, y = M @ move
    #r = np.sqrt(x*x+y*y)
    r = np.sqrt(t*t-1)
    theta = np.arctan2(y, x)
    t = min(t_max, t)
    r = min(r_max, r)
    coord[v] = t, r, theta
    
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
    
    coord[np.argsort(np.sum(adjacency_matrix, 0) + np.random.random()), 0] = np.log(N*np.arange(1, N+1))
    coord[:, 1] = 2 * np.pi * np.random.rand(N)
    
    coord = np.stack([np.cosh(coord[:, 0]), np.sinh(coord[:, 0]), coord[:, 1]]).T
    t_max, r_max = (N*N+1/N/N)/2, (N*N-1/N/N)/2

    log = np.empty(steps+1)
    grs = greedy_routing_score(adjacency_matrix, coord, N, ls, num_grs_iter)
    log[0] = grs

    temperature = 1 / np.arange(1, steps+1)
    if verbose:
        temperature = tqdm(temperature)

    # annealing iteration
    for step, temp in enumerate(temperature, start=1):
        
        # compute the new greedy routing score
        new_coord = random_move(coord.copy(), scale=1, t_max=t_max, r_max=r_max, adjacency_matrix=adjacency_matrix)      
        new_grs = greedy_routing_score(adjacency_matrix, new_coord, N, ls, num_grs_iter)
        
        # accept or reject the proposed change(s)
        dE = new_grs - grs
        if 0 < dE or np.random.rand() < np.exp(dE / temp):
            coord, grs = new_coord, new_grs
        log[step] = grs
    
    return log, coord