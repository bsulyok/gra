from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, truncnorm
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import networkx as nx
import warnings


organic_graphs = [
    'jazz',
    'copperfield',
    'polbooks',
    'football',
    'metabolic',
    'unicodelang',
    'crime',
    'euroroads',
    'a_song_of_ice_and_fire',
    'pdzbase',
    'netscience'
]
temporary_experiments_root = Path('/home/bsulyok/documents/projects/gra/experiments/tmp')
graphs_root = Path('/home/bsulyok/documents/projects/gra/graphs')


@dataclass
class State:
    coords: np.ndarray
    distance_matrix: np.ndarray
    closest_neighbour: np.ndarray
    destination: np.ndarray
    greedy_routing_score: float


class StateGenerator:
    def __init__(self, G: nx.Graph):
        self.G = G

        # some variables for utility
        self.N = len(self.G)
        self.num_gr_iter = int(np.ceil(np.log2(self.N-1)))
        self.arange = np.arange(self.N)
        self.num_edges = self.G.size()
        self.num_nonadjacent_vertices = self.N * (self.N-1) - 2 * self.num_edges

        # initialize the greedy arrays
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'adjacency_matrix will return a scipy.sparse array')
            self.adjm = nx.adjacency_matrix(G).toarray()
        self.adjl = np.array([np.where(row)[0] for row in self.adjm], dtype=object)
        spl_dict = dict(nx.all_pairs_shortest_path_length(G))
        self.spl = np.array([[spl_dict[u][v] for u in range(self.N)] for v in range(self.N)])

    def get_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        r, theta = coords[:, [0]], coords[:, [1]]
        return np.cosh(r)*np.cosh(r.T)-np.sinh(r)*np.sinh(r.T)*np.cos(np.pi-np.abs(np.pi-np.abs(theta-theta.T)))

    def get_closest_neighbours(self, distance: np.ndarray) -> np.ndarray:
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

    def get_greedy_routing_score(self, destination: np.ndarray) -> float:
        return (np.sum(destination == self.arange) - self.N - self.num_edges) / self.num_nonadjacent_vertices

    def get_initial_state(self, coords):
        distance_matrix = self.get_distance_matrix(coords)
        closest_neighbour = self.get_closest_neighbours(distance_matrix)
        destination = self.get_destination(closest_neighbour)
        greedy_routing_score = self.get_greedy_routing_score(destination)
        return State(coords, distance_matrix, closest_neighbour, destination, greedy_routing_score)

    def get_new_state(self, state):
        return state
        moved_vertex = self.select_vertex()
        coords = self.move_vertex(state.coords, moved_vertex)
        distance_matrix = self.recalc_distance_matrix(coords, moved_vertex)
        closest_neighbour = self.recalc_closest_neighbours(distance_matrix, moved_vertex)
        destination = self.recalc_destination(closest_neighbour, moved_vertex)
        greedy_routing_score = self.get_greedy_routing_score(destination)
        return State(coords, distance_matrix, closest_neighbour, destination, greedy_routing_score)

    def select_vertex(self) -> int:
        return np.random.randint(self.graph.size)

    def move_vertex(self, coords, vertex) -> np.ndarray:
        new_coords = coords.copy()
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = coords[vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        new_coords[vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        new_coords[vertex, 1] = np.random.normal(loc=coords[vertex, 1], scale=np.pi/4) % (2*np.pi)
        return new_coords

    def recalc_distance(self, new_coords, vertex) -> np.ndarray:
        distance = self.distm.copy()
        r, theta = new_coords.T
        d_theta = np.pi - np.abs(np.pi - np.abs(theta[vertex]-theta))
        dist = np.cosh(r[vertex])*np.cosh(r)-np.sinh(r[vertex])*np.sinh(r)*np.cos(d_theta)
        dist[vertex] = 1
        distance[vertex] = distance[:, vertex] = dist
        return distance

    def recalc_greedy_closest_neighbour(self, distance, vertex):
        cn = self.cn.copy()
        neighbours = self.adjl[vertex]
        cn[neighbours] = np.array([neigh[np.argmin(distance[neigh], axis=0)] for neigh in self.adjl[neighbours]])
        cn[:, vertex] = np.argmin(self.madjm + distance[vertex], axis=1)
        np.fill_diagonal(cn, self.arange)
        return cn

    def recalc_greedy_destination(self, gcn, vertex):
        gd = self.gd.copy()
        neighbours = self.adjl[vertex]
        changed = np.any(gcn[neighbours] != self.cn[neighbours], axis=0)
        changed[vertex] = True
        gd_changed = gcn[:, changed]
        for _ in range(self.num_gr_iter):
            gd_next_hop = gd_changed[gd_changed, self.arange]
            gd_changed = gd_next_hop
        gd[:, changed] = gd_changed
        return gd

    def get_greedy_routing_score(self, destination: np.ndarray) -> float:
        return (np.sum(destination == self.arange) - self.N - self.num_edges) / self.num_nonadjacent_vertices

    def get_metrics(self) -> dict:
        metrics = {}

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in arccosh')
            distance = np.arccosh(self.distm)
        np.fill_diagonal(distance, 0.0)

        metrics['grs'] = self.grs
        metrics['auroc'] = roc_auc_score(self.adjm.astype(int).flatten(), -distance.flatten())
        metrics['auprc'] = average_precision_score(self.adjm.astype(int).flatten(), -distance.flatten())

        u, v = np.triu_indices(self.spl.shape[0], k=1)
        metrics['mapping_accuracy'] = spearmanr(distance[u, v], self.spl[u, v]).correlation

        source = vertex = np.tile(self.arange, self.N)
        target = np.repeat(self.arange, self.N)
        greedy_path_length = np.full_like(self.cn, np.iinfo(self.cn.dtype).max)
        for length in range(self.N):
            arrived = vertex == target
            greedy_path_length[source[arrived], target[arrived]] = length
            source, vertex, target = source[~arrived], vertex[~arrived], target[~arrived]
            vertex = self.cn[vertex, target]
        u, v = np.where(1 - (self.adjm + np.eye(self.N)))
        metrics['weighted_grs'] = np.mean(self.spl[u, v] / greedy_path_length[u, v])

        greedy_path_distance = np.full_like(self.distm, 0.0)
        source = vertex = np.tile(self.arange, self.N)
        target = np.repeat(self.arange, self.N)
        for _ in range(self.N):
            arrived = vertex == target
            source, vertex, target = source[~arrived], vertex[~arrived], target[~arrived]
            next_vertex = self.cn[vertex, target]
            greedy_path_distance[source, target] += distance[vertex, next_vertex]
            vertex = next_vertex
        u, v = np.where(1 - (self.adjm + np.eye(self.N)))
        greedy_path_distance[source, target] = np.finfo(self.distm.dtype).max
        u, v = np.where(1 - (self.adjm + np.eye(self.N)))
        metrics['greedy_routing_efficiency'] = np.mean(distance[u, v] / greedy_path_distance[u, v])

        return metrics

RandomSampling = StateGenerator