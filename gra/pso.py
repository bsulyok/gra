import numpy as np
from networkx import Graph


def stochastic_rounding(p: float) -> int:
    floor = int(np.floor(p))
    if np.random.rand() < p - floor:
        return floor+1
    else:
        return floor
    

def popularity_similarity_optimisation_model(
    N: int,
    m: int,
    beta: float = 0.5,
    T: float = 0.5,
    radial_coord: np.ndarray = None,
    angular_coord: np.ndarray = None,
    force_connectivity: bool = False
) -> Graph:

    def connection_probability(r):
        return 1 / (1+np.exp(r/2/T))

    if beta == 0:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*(1-np.exp(-r/2))/m)
    elif beta == 1:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*r/m)
    else:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*(1-np.exp(-r*(1-beta)/2))/m/(1-beta))

    G = Graph()    

    if radial_coord is None:
        radial_coord = 2 * np.log(np.arange(1, N+1))
    if angular_coord is None:
        angular_coord = 2 * np.pi * np.random.random(N)

    for idx_new in range(N):
        final_radial_coord = beta * radial_coord[-1] + (1-beta) * radial_coord[idx_new]
        r_vertex, theta_vertex = final_radial_coord, angular_coord[idx_new]
        G.add_node(str(idx_new), coords=[r_vertex, theta_vertex])

        if idx_new <= m:
            neighbours = np.arange(idx_new)
        else:
            r_old = beta * radial_coord[idx_new] + (1-beta) * radial_coord[:idx_new]
            theta_old = angular_coord[:idx_new]
            r_new = radial_coord[idx_new]
            theta_new = angular_coord[idx_new]
            with np.errstate(divide='ignore', invalid='ignore'):
                ang_diff = np.pi - np.abs(np.pi - np.abs(theta_new - theta_old))
                vertex_distance = np.cosh(r_new) * np.cosh(r_old)
                vertex_distance -= np.sinh(r_new) * np.sinh(r_old) * np.cos(ang_diff)
                vertex_distance = np.arccosh(vertex_distance)
            if T == 0:
                neighbours = sorted(range(idx_new), key=lambda idx_old: vertex_distance[idx_old])[:m]
            else:
                cutoff = cutoff_function(r_old)
                conn_prob = connection_probability(vertex_distance-cutoff)
                if force_connectivity:
                    neighbours = np.random.choice(idx_new, stochastic_rounding(m), replace=False, p=conn_prob/sum(conn_prob))
                else:
                    neighbours = np.where(np.random.rand(idx_new) < conn_prob)[0]
        for idx_old in neighbours:
            G.add_edge(str(idx_new), str(idx_old))

    return G


PSO = popularity_similarity_optimisation_model