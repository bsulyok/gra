from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, truncnorm
from dataclasses import dataclass
import numpy as np
import networkx as nx
import warnings
from scipy import sparse as sp


@dataclass
class State:
    coords: np.ndarray
    distance_matrix: np.ndarray
    closest_neighbour: np.ndarray
    destination: np.ndarray
    success_rate: float


class StateGenerator:
    def __init__(self, adjm: np.array):
        self.adjm = adjm

        # some variables for utility
        self.N = len(self.adjm)
        self.arange = np.arange(self.N)
        self.num_gr_iter = int(np.ceil(np.log2(self.N-1)))

        # initialize the greedy arrays
        self.madjm = ~self.adjm * np.finfo(np.float64).max
        self.adjl = np.array([np.where(row)[0] for row in self.adjm], dtype=object)
        self.spl = sp.csgraph.shortest_path(self.adjm)

    def get_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        r, theta = coords[:, [0]], coords[:, [1]]
        return np.cosh(r)*np.cosh(r.T)-np.sinh(r)*np.sinh(r.T)*np.cos(np.pi-np.abs(np.pi-np.abs(theta-theta.T)))

    def get_closest_neighbour(self, distance: np.ndarray) -> np.ndarray:
        closest_neighbour = np.empty_like(distance, dtype=int)
        for vertex, neighbours in enumerate(self.adjl):
            closest_neighbour[vertex] = neighbours[np.argmin(distance[neighbours], axis=0)]
        np.fill_diagonal(closest_neighbour, np.arange(closest_neighbour.shape[0]))
        return closest_neighbour

    def get_destination(self, closest_neighbour: np.ndarray) -> np.ndarray:
        destination = closest_neighbour.copy()
        for _ in range(self.num_gr_iter):
            destination = destination[destination, self.arange]
        return destination

    def get_success_rate(self, destination: np.ndarray) -> float:
        return (np.sum(destination == self.arange) - self.N) / self.N / (self.N-1)

    def get_initial_state(self, coords):
        distance_matrix = self.get_distance_matrix(coords)
        closest_neighbour = self.get_closest_neighbour(distance_matrix)
        destination = self.get_destination(closest_neighbour)
        success_rate = self.get_success_rate(destination)
        return State(coords, distance_matrix, closest_neighbour, destination, success_rate)

    def get_new_state(self, state):
        moved_vertex = self.select_vertex(state)
        coords = self.move_vertex(state, moved_vertex)
        distance_matrix = self.recalc_distance_matrix(state, coords, moved_vertex)
        closest_neighbour = self.recalc_closest_neighbour(state, distance_matrix, moved_vertex)
        destination = self.recalc_destination(state, closest_neighbour, moved_vertex)
        success_rate = self.get_success_rate(destination)
        return State(coords, distance_matrix, closest_neighbour, destination, success_rate)

    def select_vertex(self, state) -> int:
        return np.random.randint(self.N)

    def move_vertex(self, state, moved_vertex) -> np.ndarray:
        new_coords = state.coords.copy()
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = state.coords[moved_vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        new_coords[moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        new_coords[moved_vertex, 1] = np.random.normal(loc=state.coords[moved_vertex, 1], scale=np.pi/4) % (2*np.pi)
        return new_coords

    def recalc_distance_matrix(self, state, coords, moved_vertex) -> np.ndarray:
        distance = state.distance_matrix.copy()
        r, theta = coords.T
        d_theta = np.pi - np.abs(np.pi - np.abs(theta[moved_vertex]-theta))
        dist = np.cosh(r[moved_vertex])*np.cosh(r)-np.sinh(r[moved_vertex])*np.sinh(r)*np.cos(d_theta)
        dist[moved_vertex] = 1
        distance[moved_vertex] = distance[:, moved_vertex] = dist
        return distance

    def recalc_closest_neighbour(self, state, distance, moved_vertex):
        closest_neighbour = state.closest_neighbour.copy()
        neighbours = self.adjl[moved_vertex]
        closest_neighbour[neighbours] = np.array([neigh[np.argmin(distance[neigh], axis=0)] for neigh in self.adjl[neighbours]])
        closest_neighbour[:, moved_vertex] = np.argmin(self.madjm + distance[moved_vertex], axis=1)
        np.fill_diagonal(closest_neighbour, self.arange)
        return closest_neighbour

    def recalc_destination(self, state, closest_neighbour, moved_vertex):
        destination = state.destination.copy()
        neighbours = self.adjl[moved_vertex]
        changed = np.any(state.closest_neighbour[neighbours] != closest_neighbour[neighbours], axis=0)
        changed[moved_vertex] = True
        gd_changed = closest_neighbour[:, changed].copy()
        arange = np.arange(gd_changed.shape[1])
        for _ in range(self.num_gr_iter):
            gd_changed = gd_changed[gd_changed, arange]
        destination[:, changed] = gd_changed
        return destination

    def get_destination(self, closest_neighbour: np.ndarray) -> np.ndarray:
        destination = closest_neighbour.copy()
        for _ in range(self.num_gr_iter):
            destination = destination[destination, self.arange]
        return destination

    def get_metrics(self, state) -> dict:
        metrics = {}

        #with warnings.catch_warnings():
        #    warnings.filterwarnings('ignore', 'invalid value encountered in arccosh')
        #    distm = np.arccosh(state.distance_matrix)
        #distm[np.isnan(distm)] = np.finfo(distm.dtype).min

        distm = np.arccosh(state.distance_matrix.clip(1, None))

        u, v = np.triu_indices(self.spl.shape[0], k=1)

        pair_distances = distm[u, v]
        pair_connections = self.adjm[u, v]

        metrics['success_rate'] = state.success_rate
        metrics['auroc'] = roc_auc_score(1-pair_connections.astype(int), pair_distances)
        #metrics['auprc'] = average_precision_score(1-pair_connections.astype(int), pair_distances)
        metrics['auprc'] = average_precision_score(pair_connections.astype(int), 1/pair_distances)

        u, v = np.triu_indices(self.spl.shape[0], k=1)
        metrics['mapping_accuracy'] = spearmanr(distm[u, v], self.spl[u, v]).correlation

        source = vertex = np.tile(self.arange, self.N)
        target = np.repeat(self.arange, self.N)
        greedy_path_length = np.full_like(state.closest_neighbour, np.iinfo(state.closest_neighbour.dtype).max)
        for length in range(self.N):
            arrived = vertex == target
            greedy_path_length[source[arrived], target[arrived]] = length
            source, vertex, target = source[~arrived], vertex[~arrived], target[~arrived]
            vertex = state.closest_neighbour[vertex, target]
        u, v = np.where(1 - np.eye(self.N))
        metrics['greedy_routing_score'] = np.mean(self.spl[u, v] / greedy_path_length[u, v])

        greedy_path_distance = np.full_like(distm, 0.0)
        source = vertex = np.tile(self.arange, self.N)
        target = np.repeat(self.arange, self.N)
        for _ in range(self.N):
            arrived = vertex == target
            source, vertex, target = source[~arrived], vertex[~arrived], target[~arrived]
            next_vertex = state.closest_neighbour[vertex, target]
            greedy_path_distance[source, target] += distm[vertex, next_vertex]
            vertex = next_vertex
        u, v = np.where(~(np.eye(self.N, dtype=bool) | self.adjm))
        greedy_path_distance[source, target] = np.finfo(distm.dtype).max
        metrics['greedy_routing_efficiency'] = np.mean(distm[u, v] / greedy_path_distance[u, v])

        path_count = np.eye(self.N, dtype=int)
        path_sum = np.zeros((self.N, self.N), dtype=float)
        visited = np.eye(self.N, dtype=bool)
        next_level = np.eye(self.N, dtype=bool)
        while not np.all(visited):
            level, next_level = next_level, np.zeros((self.N, self.N), dtype=bool)
            for source, vertex in zip(*np.where(level)):
                for neighbour in self.adjl[vertex]:
                    if not visited[source, neighbour]:
                        next_level[source, neighbour] = True
                        path_count[source, neighbour] += path_count[source, vertex]
                        path_sum[source, neighbour] += path_sum[source, vertex] + distm[vertex, neighbour] * path_count[source, vertex]
            visited |= next_level
        u, v = np.where(~(np.eye(self.N, dtype=bool) | self.adjm))
        geometrical_congruence = (distm[u, v] / (path_sum[u, v] / path_count[u, v])).mean()
        metrics['geometrical_congruence'] = geometrical_congruence

        return metrics


#######################################################################################################################


class RandomSampling(StateGenerator):
    def __init__(self, **args):
        super().__init__(**args)


#######################################################################################################################


class DegreeDependentSampling(StateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
        degree = self.adjm.sum(axis=0)
        self.sampling_probability = degree / np.sum(degree)
    
    def select_vertex(self, state) -> int:
        return np.random.choice(self.N, p=self.sampling_probability)


#######################################################################################################################


class SourceDependentSampling(StateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_vertex(self, state) -> int:
        unsuccessful_paths = np.sum(state.destination != self.arange, axis=1)
        unsuccessful_paths += 1
        return np.random.choice(self.N, p=unsuccessful_paths / np.sum(unsuccessful_paths))


#######################################################################################################################


class TargetDependentSampling(StateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_vertex(self, state) -> int:
        unsuccessful_paths = np.sum(state.destination != self.arange, axis=0)
        unsuccessful_paths += 1
        return np.random.choice(self.N, p=unsuccessful_paths / np.sum(unsuccessful_paths))


__all__ = [
    'RandomSampling',
    'DegreeDependentSampling',
    'SourceDependentSampling',
    'TargetDependentSampling'
]