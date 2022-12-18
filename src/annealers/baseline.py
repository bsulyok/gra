import numpy as np
from dataclasses import dataclass
import sys
sys.path.append('../')
from typing import Callable, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
import time
import uuid
import yaml
from ..utils.pso import PSO
from ..utils.graph import Graph
from ..utils.embedding import mercator_embedding, random_embedding, stupid_random_embedding
from ..utils.functions import initial_distance, initial_greedy_closest_neighbour, initial_greedy_destination, greedy_routing_score
import networkx as nx
import warnings
import csv
from tqdm import tqdm
from scipy import stats
from itertools import combinations, pairwise


organic_graphs = [
    'football',
    'metabolic',
    'unicodelang',
    'crime'
]
temporary_experiments_root = Path('/home/bsulyok/documents/projects/gra/experiments/tmp')
graphs_root = Path('/home/bsulyok/documents/projects/gra/graphs')

@dataclass
class State:
    coords: np.ndarray
    distance: np.ndarray
    gcn: np.ndarray
    gd: np.ndarray
    grs: float

@dataclass
class GRAExperiment:
    def __init__(self, name: str, path: Path, graph: Graph, state: State, step: int, time: float):
        self.name = name
        self.path = path
        self.graph = graph
        self.state = state
        self.step = step
        self.time = time    
        self.madjm = ~self.graph.adjm * np.finfo(np.float64).max
        self.N = self.graph.size
        self.num_grs_iter = int(np.ceil(np.log2(self.N-1)))
        self.volume = self.graph.adjm.sum()
        self.volume_nonadjacent = self.N * (self.N-1) - self.volume
        self.G = nx.Graph()
        self.G.add_edges_from(self.graph.edges)
        self.shortest_paths = {}
        for u, v in combinations(range(self.N), 2):
            if not self.graph.adjm[u, v]:
                self.shortest_paths[(u, v)] = list(nx.all_shortest_paths(self.G, u, v))
        shortest_path_lengths = dict(nx.shortest_path_length(self.G))
        self.spl = np.array([[shortest_path_lengths[u][v]for v in range(self.N)] for u in range(self.N)])
        self.arange = np.arange(self.N)
            
    @staticmethod
    def new(graph_name: str, embedding_name: str = 'random', path: Optional[Path] = None):
        if path is None:
            name = uuid.uuid4().hex
            path = temporary_experiments_root / name
        else:
            name = path.stem
        # create graph
        if graph_name in organic_graphs:
            graph = Graph.load(graphs_root / graph_name)
        else:
            graph = Graph.new(graph_name)
        # create initial embedding
        if embedding_name == 'mercator':
            coords = mercator_embedding(graph)
        elif embedding_name == 'random':
            coords = random_embedding(graph)
        else:
            raise 'this embedding method is not implemented'
        # create initial state
        distance = initial_distance(coords)
        gcn = initial_greedy_closest_neighbour(distance, graph.adjl)
        gd = initial_greedy_destination(gcn)
        grs = greedy_routing_score(gd)
        state = State(coords, distance, gcn, gd, grs)
        step = 0
        time = 0.0
        # save exp
        path.mkdir(parents=True)
        conf = {
            'graph_name': graph.name,
            'size': int(graph.size),
            'annealer_name': 'baseline',
            'embedding_name': embedding_name,
            'grs': float(grs),
            'step': step,
            'time': time,
        }
        with open(path / 'conf.yaml', 'w') as conf_file:
            yaml.safe_dump(conf, conf_file)
        np.savetxt(path / 'edges.csv', graph.edges, delimiter=',', fmt='%i')
        np.savetxt(path / 'initial_coords.csv', coords, delimiter=',')
        return GRAExperiment(name, path, graph, state, step, time)

    @staticmethod
    def load(path: Path):
        name = path.stem
        graph = Graph.load(path)
        with open(path / 'conf.yaml', mode='r') as conf_file:
            conf = yaml.safe_load(conf_file)
        try:
            coords = np.loadtxt(path / 'coords.csv', delimiter=',')
        except:
            coords = np.loadtxt(path / 'initial_coords.csv', delimiter=',')
        distance = initial_distance(coords)
        gcn = initial_greedy_closest_neighbour(distance, graph.adjl)
        gd = initial_greedy_destination(gcn)
        grs = greedy_routing_score(gd)
        state = State(coords, distance, gcn, gd, grs)
        step = conf['step']
        time = conf['time']
        return GRAExperiment(name, path, graph, state, step, time)
    
    def write_logs(self):
        metrics = {
            'step': int(self.step),
            'time': float(self.time),
            'grs': float(self.state.grs)
        }
        metrics.update(self.get_metrics())
        logs_path = self.path / 'logs.csv'
        if not logs_path.exists():
            with open(logs_path, 'w') as logs_file:
                csv.writer(logs_file, delimiter=',').writerow(metrics.keys())
        with open(logs_path, 'a') as logs_file:
            csv.writer(logs_file, delimiter=',').writerow(metrics.values())
            
        conf_path = self.path / 'conf.yaml'
        with open(conf_path, 'r') as conf_file:
            conf = yaml.safe_load(conf_file)
        conf.update(metrics)
        with open(conf_path, 'w') as conf_file:
            yaml.safe_dump(conf, conf_file)

    def get_metrics(self) -> dict:
        metrics = {}
        
        # nonadjacent grs
        successful = np.sum(self.state.gd == self.arange)
        grs_nonadjacent = (successful - self.N - self.volume) / self.volume_nonadjacent
        metrics['grs_nonadjacent'] = float(grs_nonadjacent)
        
        # mapping accuracy
        dist, spl = [], []
        for source, target in combinations(range(self.N), 2):
            spl.append(self.spl[source][target])
            dist.append(self.state.distance[source, target])
        mapping_accuracy = stats.spearmanr(spl, dist).correlation
        metrics['mapping_accuracy'] = float(mapping_accuracy)

        # geometrical congruence
        gc_score = 0
        for (u, v), shortest_paths in self.shortest_paths.items():
            ptsp = np.mean([sum(self.state.distance[i, j] for i, j in pairwise(path)) for path in shortest_paths])
            dist = self.state.distance[u, v]
            gc_score += dist / ptsp
        geometrical_congruence = gc_score / self.volume_nonadjacent
        metrics['geometrical_congruence'] = float(geometrical_congruence)  
        
        # alernative grs
        N = self.N
        greedy_routing_distance = np.zeros_like(self.spl)
        gd = np.arange(self.N).repeat(self.N).reshape(self.N, self.N)
        for _ in range(self.N):
            greedy_routing_distance[gd != self.arange] += 1
            gd = self.state.gcn[gd, self.arange]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            path_ratios = self.spl / greedy_routing_distance
        np.fill_diagonal(path_ratios, 0)
        path_ratios[greedy_routing_distance == self.N] = 0
        grs_alt = path_ratios.sum() / self.volume_nonadjacent
        metrics['grs_alt'] = float(grs_alt)
        
        return metrics

    def write(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)
            self.graph.write(self.path)
        self.state.write(self.path, self.get_metrics())

    def embed(self, steps: int, log_freq: Optional[int] = None):
        log_freq = self.graph.size if log_freq is None else log_freq
        if self.step == 0:
            self.write_logs()
        start = time.time()
        for step in tqdm(range(1, int(steps)+1)):
            self.state = self.step_function(self.state)
            self.step += 1
            if step % log_freq == 0:
                self.time += time.time() - start
                self.write_logs()
                start = time.time()

    def step_function(self, state):
        vertex = self.select_vertex()
        coords = self.move_vertex(vertex)
        distance = self.recalc_distance(coords, vertex)
        gcn = self.recalc_greedy_closest_neighbour(distance, vertex)
        gd = self.recalc_greedy_destination(gcn, vertex)
        grs = self.greedy_routing_score(gd)
        dE = self.state.grs - grs
        temp = 1 / (self.step * self.graph.size + 1)
        self.step += 1
        if dE < 0 or np.random.rand() < np.exp(-dE / temp):
            return State(coords, distance, gcn, gd, grs)
        else:
            return state

    def select_vertex(self) -> int:
        return np.random.randint(self.graph.size)

    def move_vertex(self, vertex) -> np.ndarray:
        N = self.graph.size
        coords = self.state.coords.copy()
        clip_min, clip_max = 1, 2*np.log(N)
        loc, scale = self.state.coords[vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        coords[vertex, 0] = stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        coords[vertex, 1] = np.random.normal(loc=self.state.coords[vertex, 1], scale=np.pi/4) % (2*np.pi)
        return coords
    
    def recalc_distance(self, coords, vertex) -> np.ndarray:
        distance = self.state.distance.copy()
        r, theta = coords.T
        d_theta = np.pi - np.abs(np.pi - np.abs(theta[vertex]-theta))
        dist = np.cosh(r[vertex])*np.cosh(r)-np.sinh(r[vertex])*np.sinh(r)*np.cos(d_theta)
        dist[vertex] = 1
        distance[vertex] = distance[:, vertex] = dist
        return distance
    
    def recalc_greedy_closest_neighbour(self, distance, vertex):
        gcn = self.state.gcn.copy()
        neighbours = self.graph.adjl[vertex]
        gcn[neighbours] = np.array([neigh[np.argmin(distance[neigh], axis=0)] for neigh in self.graph.adjl[neighbours]])
        gcn[:, vertex] = np.argmin(self.madjm + distance[vertex], axis=1)
        np.fill_diagonal(gcn, np.arange(self.graph.size))
        return gcn
    
    def recalc_greedy_destination(self, gcn, vertex):
        gd = self.state.gd.copy()
        neighbours = self.graph.adjl[vertex]
        changed = np.any(gcn[neighbours] != self.state.gcn[neighbours], axis=0)
        changed[vertex] = True
        gd_changed = gcn[:, changed]
        arange = np.arange(gd_changed.shape[1])
        num_grs_iter = int(np.ceil(np.log2(self.graph.size-1)))
        for _ in range(self.num_grs_iter):
            gd_next_hop = gd_changed[gd_changed, arange]
            gd_changed = gd_next_hop
        #while np.any((gd_next_hop:=gd_changed[gd_changed, arange]) != gd_changed):
        #    gd_changed = gd_next_hop
        #    print(gd_changed.shape)
        gd[:, changed] = gd_changed
        return gd

    def greedy_routing_score(self, gd: np.ndarray) -> float:
        N = self.graph.size
        successful = np.sum(gd == np.arange(N))
        return (successful - N) / N / (N-1)