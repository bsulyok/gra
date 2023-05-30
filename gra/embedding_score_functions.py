import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from scipy import sparse as sp
from typing import Callable
from scipy.sparse import spmatrix
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse.csgraph import shortest_path

from .embedding_state_generators import EmbeddingState
from .geometry import native_disk_distance


__all__ = [
    'EmbeddingState',
    'EmbeddingScoreFunction',
    'GeometricalCongruence',
    'GreedyRoutingSuccessRate',
    'GreedyRoutingScore',
    'GreedyRoutingEfficiency',
    'EdgePredictionAUROC',
    'EdgePredictionAUPRC',
    'MappingAccuracy'
]


class EmbeddingScoreFunction(Callable):
    def __init__(self):
        pass

    def compute_score(self, state: EmbeddingState, update: bool) -> float:
        raise NotImplementedError

    def recompute_score(self, state: EmbeddingState, previous_state: EmbeddingState, update: bool) -> float:
        raise NotImplementedError
    
    def initial_state(self, coords: np.ndarray) -> EmbeddingState:
        state = EmbeddingState(coords=coords)
        score = self.compute_score(state, update=True)
        return state
    
    def __call__(self, state: EmbeddingState, previous_state: EmbeddingState = None, update: bool = False) -> float:
        if previous_state is None:
            score = self.compute_score(state=state, update=update)
        else:
            score = self.recompute_score(state=state, previous_state=previous_state, update=update)
        return score


class GeometricalCongruence(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        self.spadjm = spadjm
        N = self.spadjm.shape[0]
        self.distance_function = distance_function
        spadjm_coo = self.spadjm.tocoo()
        self.tsp_row = spadjm_coo.row
        self.tsp_col = spadjm_coo.col
        indices = ~np.eye(N, dtype=bool)
        if exclude_neighbours:
            indices[spadjm_coo.row, spadjm_coo.col] = False
        self.indices = indices.flatten()
        
        M = sp.csr_array((N*N, N*N))
        path_count = np.eye(N, dtype=int).flatten()
        next_level = np.arange(N) * (N+1)
        while next_level.size:
            source, target = next_level // N, next_level % N
            asd = self.spadjm[source].tocoo()
            s = source[asd.row]
            t = target[asd.row]
            n = asd.col
            unvisited = path_count[t*N+n] == 0
            s, t, n = s[unvisited], t[unvisited], n[unvisited]
            np.add.at(path_count, t*N+n, path_count[t*N+s])
            next_level = np.unique(t*N+n)
            i = (t*N+n).tolist()
            j = (s*N+n).tolist()
            d = (path_count[t*N+s]).tolist()
            asd2 = M[t*N+s].tocoo()
            i += (t*N+n)[asd2.row].tolist()
            j += asd2.col.tolist()
            d += asd2.data.tolist()
            M += sp.csr_matrix((d, (i, j)), shape=M.shape)
        M = M * sp.csr_matrix(1/path_count).T
        self.M = M

    def compute_score(self, state: EmbeddingState, update: bool) -> None:
        distance = self.distance_function(state.coords, state.coords[:, None])
        data = distance[self.tsp_row, self.tsp_col]
        spadjm = sp.csr_matrix((data, (self.tsp_row, self.tsp_col)), self.spadjm.shape)
        ptsp = self.M @ spadjm.toarray().flatten()
        score = np.mean(distance.flatten()[self.indices] / ptsp[self.indices])

        if update:
            state.distance = distance
            state.score = score

        return score
    
    def recompute_score(self, state: EmbeddingState, previous_state: EmbeddingState, update: bool) -> None:
        moved_vertices = np.where(np.any(state.coords != previous_state.coords, axis=1))[0]
        distance = previous_state.distance.copy()
        mv_dist = self.distance_function(state.coords[moved_vertices], state.coords[:, None])
        distance[moved_vertices] = mv_dist.T
        distance[:, moved_vertices] = mv_dist
        data = distance[self.tsp_row, self.tsp_col]
        spadjm = sp.csr_matrix((data, (self.tsp_row, self.tsp_col)), self.spadjm.shape)
        ptsp = self.M @ spadjm.toarray().flatten()
        score = score = np.mean(state.distance.flatten()[self.indices] / ptsp[self.indices])

        if update:
            state.distance = distance
            state.score = score
        
        return score


class GreedyRoutingSuccessRate(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        self.spadjm = spadjm
        
        self.N = self.spadjm.shape[0]
        self.distance_function = distance_function
        spadjm_coo = self.spadjm.tocoo()
        self.indices = ~np.eye(self.N, dtype=bool)
        if exclude_neighbours:
            self.indices[spadjm_coo.row, spadjm_coo.col] = False

        self.col = np.arange(self.N)
        self.num_iter = int(np.ceil(np.log2(self.N-1)))
        self.adjl = np.array([row.indices for row in self.spadjm], dtype=object)

        self.madjm = np.full((self.N, self.N), np.inf)
        self.madjm[spadjm.toarray().astype(bool)] = 0.0

    def compute_score(self, state: EmbeddingState, update: bool) -> None:
        distance = self.distance_function(state.coords, state.coords[:, None])
        closest_neighbour = np.array([row[distance[row].argmin(axis=0)] for row in self.adjl])
        np.fill_diagonal(closest_neighbour, self.col)
        destination = closest_neighbour.copy()
        for _ in range(self.num_iter):
            destination = destination[destination, self.col]
        successful = destination == self.col
        score = np.mean(successful[self.indices])

        if update:
            state.distance = distance
            state.closest_neighbour = closest_neighbour
            state.destination = destination
            state.successful = successful
            state.score = score
        
        return score

    def recompute_score(self, state: EmbeddingState, previous_state: EmbeddingState, update: bool) -> float:
        moved_vertices = np.where(np.any(state.coords != previous_state.coords, axis=1))[0]
        neighbours = np.unique(self.spadjm[moved_vertices].tocoo().col)

        distance = previous_state.distance.copy()
        mv_dist = self.distance_function(state.coords[moved_vertices], state.coords[:, None])
        distance[moved_vertices] = mv_dist.T
        distance[:, moved_vertices] = mv_dist

        closest_neighbour = previous_state.closest_neighbour.copy()
        closest_neighbour[neighbours] = np.array([neigh[np.argmin(distance[neigh], axis=0)] for neigh in self.adjl[neighbours]])
        closest_neighbour[:, moved_vertices] = np.argmin(self.madjm + distance[moved_vertices], axis=1)[:, np.newaxis]
        np.fill_diagonal(closest_neighbour, self.col)
        
        destination = previous_state.destination.copy()
        changed = np.any(closest_neighbour[neighbours] != previous_state.closest_neighbour[neighbours], axis=0)
        changed[moved_vertices] = True
        destination_changed = closest_neighbour[:, changed].copy()
        arange = np.arange(destination_changed.shape[1])
        for _ in range(self.num_iter):
            destination_changed = destination_changed[destination_changed, arange]
        destination[:, changed] = destination_changed

        successful = destination == self.col
        score = np.mean(successful[self.indices])

        if update:
            state.distance = distance
            state.closest_neighbour = closest_neighbour
            state.destination = destination
            state.successful = successful
            state.score = score
        
        return score


class EdgePredictionAUROC(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        adjm = spadjm.toarray()
        self.row, self.col = np.triu_indices_from(adjm, k=1) 
        self.target = adjm[self.row, self.col]
        self.distance_function = distance_function

    def compute_score(self, state: EmbeddingState) -> EmbeddingState:
        distance = state.distance if state.distance is not None else self.distance_function(state.coords, state.coords[:, None])
        return roc_auc_score(self.target, -distance[self.row, self.col])
    

class EdgePredictionAUPRC(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        adjm = spadjm.toarray()
        self.row, self.col = np.triu_indices_from(adjm, k=1) 
        self.target = adjm[self.row, self.col]
        self.distance_function = distance_function

    def compute_score(self, state: EmbeddingState) -> EmbeddingState:
        distance = state.distance if state.distance is not None else self.distance_function(state.coords, state.coords[:, None])
        return average_precision_score(self.target, -distance[self.row, self.col])
    

class MappingAccuracy(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        spl = shortest_path(spadjm)
        self.row, self.col = np.triu_indices_from(spl, k=1)
        self.target = spl[self.row, self.col]
        self.distance_function = distance_function

    def compute_score(self, state: EmbeddingState) -> EmbeddingState:
        distance = state.distance if state.distance is not None else self.distance_function(state.coords, state.coords[:, None])
        return spearmanr(self.target, distance[self.row, self.col]).correlation


class GreedyRoutingScore(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        self.spadjm = spadjm
        self.indices = ~ np.eye(spadjm.shape[0], dtype=bool)
        if exclude_neighbours:
            self.indices &= ~ spadjm.toarray().astype(bool)
        self.N = self.spadjm.shape[0]
        self.distance_function = distance_function
        self.shortest_path_length = shortest_path(spadjm)[self.indices]
        self.adjl = np.array([row.indices for row in self.spadjm], dtype=object)


    def compute_score(self, state: EmbeddingState) -> EmbeddingState:
        distance = self.distance_function(state.coords, state.coords[:, None])
        closest_neighbour = np.array([row[state.distance[row].argmin(axis=0)] for row in self.adjl])
        np.fill_diagonal(closest_neighbour, np.arange(self.N))
        greedy_path_length = np.full_like(distance, np.inf)
        np.fill_diagonal(greedy_path_length, 0)

        vertex = np.arange(self.N) * (self.N+1)
        path_length = 1
        while vertex.size:
            target, source = vertex // self.N, vertex % self.N
            asd = self.spadjm[source].tocoo()
            t = target[asd.row]
            s = source[asd.row]
            v = asd.col
            not_yet_visited = greedy_path_length[t, v] == np.inf
            t, s, v = t[not_yet_visited], s[not_yet_visited], v[not_yet_visited]
            is_closest = closest_neighbour[v, t] == s
            t, s, v = t[is_closest], s[is_closest], v[is_closest]

            greedy_path_length[t, v] = path_length
            vertex = self.N * t + v

            path_length += 1
        
        return np.mean(self.shortest_path_length / greedy_path_length[self.indices])


class GreedyRoutingEfficiency(EmbeddingScoreFunction):
    def __init__(self,
        spadjm: spmatrix,
        exclude_neighbours: bool = True,
        distance_function: Callable = native_disk_distance
    ):
        self.spadjm = spadjm
        self.indices = ~ np.eye(spadjm.shape[0], dtype=bool)
        if exclude_neighbours:
            self.indices &= ~ spadjm.toarray().astype(bool)
        self.N = self.spadjm.shape[0]
        self.distance_function = distance_function
        self.adjl = np.array([row.indices for row in self.spadjm], dtype=object)


    def compute_score(self, state: EmbeddingState) -> EmbeddingState:
        distance = self.distance_function(state.coords, state.coords[:, None])
        closest_neighbour = np.array([row[state.distance[row].argmin(axis=0)] for row in self.adjl])
        np.fill_diagonal(closest_neighbour, np.arange(self.N))
        projected_greedy_path_length = np.full_like(distance, np.inf)
        np.fill_diagonal(projected_greedy_path_length, 0.0)

        vertex = np.arange(self.N) * (self.N+1)
        while vertex.size:
            target, source = vertex // self.N, vertex % self.N
            asd = self.spadjm[source].tocoo()
            t = target[asd.row]
            s = source[asd.row]
            v = asd.col
            not_yet_visited = projected_greedy_path_length[t, v] == np.inf
            t, s, v = t[not_yet_visited], s[not_yet_visited], v[not_yet_visited]
            is_closest = closest_neighbour[v, t] == s
            t, s, v = t[is_closest], s[is_closest], v[is_closest]

            projected_greedy_path_length[t, v] = projected_greedy_path_length[t, s] + distance[s, v]
            vertex = self.N * t + v
        
        return np.mean(distance[self.indices] / projected_greedy_path_length[self.indices])


class VertexHeatmapScore(EmbeddingScoreFunction):
    def __init__(self,
            target_function: EmbeddingScoreFunction,
            resolution: int = 10
        ):
        self.target_function = target_function
        self.N = self.target_function.N
        r_max = 2 * np.log(self.N)
        ls = np.linspace(-r_max, r_max, resolution)
        x, y = np.meshgrid(ls, ls)
        x, y = x.flatten(), y.flatten()
        inside_of_the_circle = np.sqrt(x*x + y*y)  < r_max+0.01
        x, y = x[inside_of_the_circle], y[inside_of_the_circle]
        self.x, self.y = x, y

    def compute_score(self, state: EmbeddingState, update: bool) -> None:
        current_score = state.score
        scores = []
        for moved_vertex in range(self.N):
            for x_, y_ in zip(self.x, self.y):
                new_coords = state.coords.copy()
                new_coords[moved_vertex] = x_, y_
                scores.append(self.target_function(EmbeddingState(coords=new_coords), state))
        score_change = np.array(scores) - current_score
        score = np.mean(0 < score_change)
        return score