from tqdm import tqdm
import numpy as np
from scipy.stats import truncnorm
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf

class BaseAnnealer:
    def __init__(self, exp_path):
        self.conf = OmegaConf.load(exp_path / 'conf.yml')
        self.exp_path = exp_path
        edges = np.loadtxt(exp_path / 'edges.csv', delimiter=',', dtype=int)
        self.N = self.conf.N
        self.adjacency_matrix = np.zeros((self.N, self.N), dtype=bool)
        self.adjacency_matrix[edges[:, 0], edges[:, 1]] = True
        self.adjacency_matrix[edges[:, 1], edges[:, 0]] = True
        self.adjacency_list = np.array([np.where(neigh)[0] for neigh in self.adjacency_matrix], dtype=object)
        self.num_grs_iter = int(np.ceil(np.log2(self.N-1)))
        self.adjacency_matrix_mask = ~self.adjacency_matrix * np.finfo(np.float64).max
        self.step = self.conf.step if self.conf.step else 1
        
        coord_path = exp_path / 'coords.csv'
        if not coord_path.exist():
            self.coord = self.generate_random_coord()
        else:
            self.coord = np.loadtxt(coord_path, delimiter=',')
        
        self.distance = self.get_distance(self.coord)
        self.gcn = self.greedy_closest_neighbour(self.distance)
        self.gd = self.greedy_destination(self.gcn)
        self.grs = self.greedy_routing_score(self.gd)

    def generate_random_coord(self):
        N = self.adjacency_matrix.shape[0]
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = 2.5*np.log(self.N), np.log(self.N) / 2
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        rng = truncnorm(a=a, b=b, loc=loc, scale=scale)
        new_r = rng.rvs(self.N)
        new_r = np.sort(new_r)[::-1]
        new_theta = 2 * np.pi * np.random.rand(self.N)
        degree_order = np.argsort(np.sum(self.adjacency_matrix, 0)+np.random.rand(self.N))
        coord = np.empty((self.N, 2))
        coord[degree_order, 0] = new_r
        coord[:, 1] = new_theta
        return coord

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
        self.coord = self.new_coord
        self.distance = self.new_distance
        self.gcn = self.new_gcn
        self.gd = self.new_gd
        self.grs = self.new_grs
    
    def embed(self, steps):
        log = []
        with open(self.exp_path / 'progress.tqdm', mode='a') as progress_file:
            for _ in tqdm(range(steps), file=progress_file, miniters=steps//100):
                self.moved_vertex = self.select_for_move()
                self.new_coord = self.random_move()
                self.new_distance = self.recalc_distance()
                self.new_gcn = self.recalc_greedy_closest_neighbour()
                self.new_gd = self.recalc_greedy_destination()
                self.new_grs = self.greedy_routing_score(self.new_gd)
                self.temp = 1 / self.step / self.N
                dE = self.energy_change()
                if dE < 0 or np.random.rand() < np.exp(-dE / self.temp):
                    self.update()
                log.append(self.grs)
                self.step += 1
        
        with open(self.exp_path / 'log.csv', mode='ba') as log_file:
            np.savetxt(log_file, log)
        np.savetxt(self.exp_path / 'coords.csv', self.coord)
    
    def ensemble_embed(self, steps, ensemble_size, **tqdm_args):
        log = []
        for _ in range(ensemble_size):
            annealer = deepcopy(self)
            log.append(annealer.embed(steps=steps, **tqdm_args))        
        return np.mean(log, axis=0), np.std(log, axis=0)