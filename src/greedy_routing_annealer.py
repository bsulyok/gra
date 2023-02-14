import numpy as np
from tqdm import tqdm
import networkx as nx

from typing import Optional, Union
import time
import csv
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import yaml
import uuid
from scipy import sparse as sp
from . import pso
from . import embedding
from . import models
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
temporary_experiments_root = Path('/home/kutato07@njmcs.local/documents/projects/gra/experiments/tmp')
graphs_root = Path('/home/kutato07@njmcs.local/documents/projects/gra/graphs')


@dataclass
class State:
    coords: np.ndarray
    distance_matrix: np.ndarray
    closest_neighbour: np.ndarray
    destination: np.ndarray
    success_rate: float


class GreedyRoutingAnnealer:
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
        self.id = self.path.name

        adjm = sp.load_npz(path / 'sparse_adjacency_matrix.npz').toarray().astype(bool)
        try:
            self.coords = np.loadtxt(self.path / 'coords.csv', delimiter=',')
        except:
            self.coords = np.loadtxt(self.path / 'initial_coords.csv', delimiter=',')
        self.conf = yaml.safe_load(open(self.path / 'conf.yaml', mode='r'))
        self.state_generator = getattr(models, self.conf['model'])(adjm=adjm)
        self.state = self.state_generator.get_initial_state(self.coords)

    @staticmethod
    def new(
        graph_name: str,
        model_name: str = 'RandomSampling',
        embedding_name: Optional[str] = None,
        root : Path = temporary_experiments_root
    ):
        name = uuid.uuid4().hex
        path = root / name
        path.mkdir(parents=True)
        
        np.random.seed(int(name, 16) % (2**32))

        if graph_name in organic_graphs:
            G = nx.read_edgelist(graphs_root / graph_name / 'edges.csv', nodetype=int)
        else:
            N = int(graph_name[4:])
            assert N <= 1024, 'Do not run with more than 1024 nodes'
            G = pso.modified_popularity_similarity_optimisation_model(N, 4, 0.5, 0.1)
            while not nx.is_connected(G):
                G = pso.modified_popularity_similarity_optimisation_model(N, 4, 0.5, 0.1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'adjacency_matrix will return a scipy.sparse array instead of a matrix')
            sp.save_npz(path / 'sparse_adjacency_matrix.npz', nx.adjacency_matrix(G))

        if embedding_name is not None:
            getattr(embedding, embedding_name)(G, inplace=True)
        coords = np.array(list(nx.get_node_attributes(G, 'coords').values()))
        np.savetxt(path / 'initial_coords.csv', coords, delimiter=',')

        conf = {
            'graph_name': graph_name,
            'embedding_name': embedding_name,
            'model': model_name,
            'step': 0,
            'time': 0.0,
        }
        yaml.safe_dump(conf, open(path / 'conf.yaml', mode='w'))
        return GreedyRoutingAnnealer(path)

    def save_experiment(self):
        metrics = {'step': self.conf['step'], 'time': self.conf['time']}
        metrics.update(self.state_generator.get_metrics(self.state))
        log_file_path = self.path / 'logs.csv'
        if not log_file_path.exists():
            csv.writer(open(log_file_path, 'w')).writerow(metrics.keys())
        csv.writer(open(log_file_path, 'a')).writerow(metrics.values())
        np.savetxt(self.path / 'coords.csv', self.state.coords, delimiter=',')
        yaml.safe_dump(self.conf, open(self.path / 'conf.yaml', mode='w'))
        
    @property
    def logs(self):
        return pd.read_csv(self.path / 'logs.csv')
    
    def embed(self, steps: int, log_freq: Optional[int] = None, silent: bool = False):
        np.random.seed(int(self.id, 16) % (2**32))
        log_freq = log_freq or len(self.state.coords)
        start = time.time()
        step = self.conf['step']
        if step == 0:
            self.save_experiment()
        step_iter = range(1, int(steps)+1)
        if not silent:
            step_iter = tqdm(step_iter)
        for _ in step_iter:
            new_state = self.state_generator.get_new_state(self.state)
            dE = self.state.success_rate - new_state.success_rate
            temp = 1 / (step * self.state_generator.N + 1)
            if dE < 0 or np.random.rand() < np.exp(-dE / temp):
                self.state = new_state
            step += 1
            if step % log_freq == 0:
                self.conf['time'] += time.time() - start
                self.conf['step'] = step
                self.save_experiment()
                start = time.time()