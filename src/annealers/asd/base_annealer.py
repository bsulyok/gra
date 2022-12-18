from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
import time
import shutil
import sys
import uuid


data_root = Path('/home/bsulyok/documents/projects/gra/graphs')
tmp_exp_path = Path('/home/bsulyok/documents/projects/gra/experiments/series/tmp')

def generate_random_coord(edges_path, save_path):
    edges = np.loadtxt(Path(edges_path), delimiter=',', dtype=int)
    N = np.max(edges)+1
    adjacency_matrix = np.zeros((N, N), dtype=bool)
    adjacency_matrix[edges[:, 0], edges[:, 1]] = True
    adjacency_matrix[edges[:, 1], edges[:, 0]] = True
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = 2.5*np.log(N), np.log(N) / 2
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    rng = truncnorm(a=a, b=b, loc=loc, scale=scale)
    new_r = rng.rvs(N)
    new_r = np.sort(new_r)[::-1]
    new_theta = 2 * np.pi * np.random.rand(N)
    degree_order = np.argsort(np.sum(adjacency_matrix, 0)+np.random.rand(N))
    coords = np.empty((N, 2))
    coords[degree_order, 0] = new_r
    coords[:, 1] = new_theta
    np.savetxt(save_path, coords, delimiter=',')

def generate_experiment(graph_name, initial_embedding='random', series_path=tmp_exp_path, annealer_name=None):
    graph_data_path = data_root / graph_name
    uid = uuid.uuid4().hex
    exp_path = series_path / uid
    exp_path.mkdir(exist_ok=False, parents=True)

    conf = OmegaConf.create()
    
    edges_path = exp_path / 'edges.csv'
    shutil.copy(graph_data_path / 'edges.csv', edges_path)
    conf.edges_path = edges_path.resolve().as_posix()
    
    coords_path = exp_path / 'coords.csv'
    if initial_embedding == 'random' or initial_embedding is None:
        generate_random_coord(edges_path, coords_path)
    else:
        shutil.copy(graph_data_path / f'{initial_embedding}.csv', coords_path)
    shutil.copy(coords_path, exp_path / 'initial_coords.csv')
    conf.coords_path = coords_path.resolve().as_posix()
    conf.initial_coords_path = coords_path.resolve().as_posix()
    conf.initial_embedding = initial_embedding
    
    conf.uid = uid
    conf.graph_size = np.loadtxt(coords_path, delimiter=',').shape[0]
    conf.time = 0.0
    conf.step = 0
    
    grs_log_path = exp_path / 'grs_log.csv'
    grs_log_path.touch()
    conf.grs_log_path = grs_log_path.resolve().as_posix()
    
    progress_file_path = exp_path / 'progress.tqdm'
    progress_file_path.touch()
    conf.progress_file_path = progress_file_path.resolve().as_posix()
    
    conf.locked = False
    if annealer_name is not None:
        conf.annealer_name = annealer_name
    
    conf_path = exp_path.joinpath('conf.yaml')
    conf.conf_path = conf_path.resolve().as_posix()
    OmegaConf.save(conf, conf.conf_path)
    
    return conf_path


class BaseAnnealer:
    def __init__(self, conf_path=None, graph_name=None, initial_embedding=None, series_path=None):
        if conf_path is None or not conf_path.exists():
            conf_path = generate_experiment(graph_name=graph_name, initial_embedding=initial_embedding, annealer_name=self.__class__.__name__, series_path=series_path)
        
        conf = OmegaConf.load(conf_path)
        
        assert self.__class__.__name__ == conf.annealer_name, 'experiment loaded into the wrong annealer'
        
        self.uid = conf.uid
        self.N = conf.graph_size
        self.step = conf.get('step', 0)
        self.time = conf.get('time', 0.0)
        edges = np.loadtxt(Path(conf.edges_path), delimiter=',', dtype=int)
        self.adjacency_matrix = np.zeros((self.N, self.N), dtype=bool)
        self.adjacency_matrix[edges[:, 0], edges[:, 1]] = True
        self.adjacency_matrix[edges[:, 1], edges[:, 0]] = True
        self.adjacency_list = np.array([np.where(neigh)[0] for neigh in self.adjacency_matrix], dtype=object)
        self.num_grs_iter = int(np.ceil(np.log2(self.N-1)))
        self.adjacency_matrix_mask = ~self.adjacency_matrix * np.finfo(np.float64).max
        if 'coords_path' not in conf:
            conf.initial_embedding = 'random'
            np.savetxt(conf.coords_path, self.generate_random_coord(), delimiter=',')
            shutil.copy(conf.coords_path, conf.coords_path.parent / 'initial_coords.csv')
        self.coord = np.loadtxt(conf.coords_path, delimiter=',')
        self.distance = self.get_distance(self.coord)
        self.gcn = self.greedy_closest_neighbour(self.distance)
        self.gd = self.greedy_destination(self.gcn)
        self.grs = self.greedy_routing_score(self.gd)
        
        
        OmegaConf.save(conf, conf.conf_path)
        self.conf = conf

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.conf.locked = False
        self._save_experiment()
    
    def _save_experiment(self, grs_log=[]):
        with open(self.conf.grs_log_path, mode='ba') as grs_log_file:
            np.savetxt(grs_log_file, grs_log, delimiter=',')
        np.savetxt(self.conf.coords_path, self.coord, delimiter=',')
        
        self.conf.step = self.step
        self.conf.time = self.time
        self.conf.grs = float(self.grs)
        OmegaConf.save(self.conf, self.conf.conf_path)

    def get_distance(self, coord: np.ndarray) -> np.ndarray:
        d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
        distance = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
        distance -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
        return distance

    def greedy_closest_neighbour(self, distance: np.ndarray) -> np.ndarray:
        gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in self.adjacency_list])
        np.fill_diagonal(gcn, np.arange(self.N))
        return gcn

    def greedy_destination(self, gcn: np.ndarray) -> np.ndarray:
        gd = self.gcn.copy()
        arange = np.arange(gd.shape[1])
        for _ in range(self.num_grs_iter):
            gd = gd[gd, arange]
        return gd
    
    def greedy_routing_score(self, gd: np.ndarray) -> float:
        successful = np.sum(gd == np.arange(self.N))
        return (successful - self.N) / self.N / (self.N-1)
    
    def select_for_move(self) -> int:
        return np.random.randint(self.N)
    
    def random_move(self) -> np.ndarray:
        coord = self.coord.copy()
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = self.coord[self.moved_vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        coord[self.moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        coord[self.moved_vertex, 1] = np.random.normal(loc=self.coord[self.moved_vertex, 1], scale=np.pi/4) % (2*np.pi)
        return coord

    def recalc_distance(self) -> np.ndarray:
        distance = self.distance.copy()
        d_theta = np.pi - np.abs(np.pi - np.abs(self.new_coord[self.moved_vertex, 1]-self.new_coord[:, 1]))
        dist = np.cosh(self.new_coord[self.moved_vertex, 0]) * np.cosh(self.new_coord[:, 0])
        dist -= np.sinh(self.new_coord[self.moved_vertex, 0]) * np.sinh(self.new_coord[:, 0]) * np.cos(d_theta)
        dist[self.moved_vertex] = 1
        distance[self.moved_vertex] = distance[:, self.moved_vertex] = dist
        return distance

    def recalc_greedy_closest_neighbour(self):
        gcn = self.gcn.copy()
        
        mv = self.moved_vertex
        neighbours = self.adjacency_list[mv]
        gcn[neighbours] = np.array([neigh[np.argmin(self.new_distance[neigh], axis=0)] for neigh in self.adjacency_list[neighbours]])
        gcn[neighbours, neighbours] = neighbours
        gcn[:, self.moved_vertex] = np.argmin(self.adjacency_matrix_mask + self.new_distance[self.moved_vertex], axis=1)
        gcn[mv, mv] = mv
        return gcn

    def recalc_greedy_destination(self):
        gd = self.gd.copy()
        neighbours = self.adjacency_list[self.moved_vertex]
        changed = np.any(self.new_gcn[neighbours] != self.gcn[neighbours], axis=0)
        changed[self.moved_vertex] = True
        gd_changed = self.new_gcn[:, changed]
        arange = np.arange(gd_changed.shape[1])
        while np.any((gd_next_hop:=gd_changed[gd_changed, arange]) != gd_changed):
            gd_changed = gd_next_hop
        gd[:, changed] = gd_changed
        return gd
        
    def energy_change(self) -> float:
        return self.grs - self.new_grs
    
    def update(self):
        dE = self.energy_change()
        if dE < 0 or np.random.rand() < np.exp(-dE / self.temp):
            self.coord = self.new_coord
            self.distance = self.new_distance
            self.gcn = self.new_gcn
            self.gd = self.new_gd
            self.grs = self.new_grs
            
    def embed(self, steps, verbose=False, log_freq=100):
        assert not self.conf.get('locked', False)
        self.conf.locked = True
        steps = int(steps)
        if self.grs == 1.0:
            return
            
        log = []
        start = time.time()
        with open(self.conf.progress_file_path, mode='a') as progress_file:
            tqdm_args = {
                'iterable': range(steps) if not verbose else tqdm(range(steps)),
                'file': progress_file,
                'miniters': steps // 100,
                'maxinterval': 1800
            }
            for _ in tqdm(**tqdm_args):
                self.step += 1
                self.moved_vertex = self.select_for_move()
                self.new_coord = self.random_move()
                self.new_distance = self.recalc_distance()
                self.new_gcn = self.recalc_greedy_closest_neighbour()
                self.new_gd = self.recalc_greedy_destination()
                self.new_grs = self.greedy_routing_score(self.new_gd)
                self.temp = 1 / self.step / self.N
                self.update()
                log.append(self.grs)
                if self.grs == 1:
                    break
                if self.step % log_freq == 0:
                    self._save_experiment(log)
        self.time += time.time() - start
        self.conf.locked = False
        self._save_experiment(log)
        
    def grs_log(self):
        log = np.loadtxt(self.conf.grs_log_path, delimiter=',')
        return log