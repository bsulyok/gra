import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
import ipycytoscape
from plotly import graph_objects as go
import warnings

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, truncnorm
from collections import defaultdict
from operator import itemgetter
from scipy import sparse as sp
from itertools import combinations, pairwise
from typing import Optional
import time
from dataclasses import dataclass
from pathlib import Path
import yaml
import uuid
from math import inf
import shutil
from . import embedding
from .state_generators.state_generator import StateGenerator

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

class GreedyRoutingAnnealer:
    def __init__(self, path: Path):
        self.path = path

        self.G = nx.read_edgelist(self.path / 'edges.csv', nodetype=int)
        try:
            self.coords = np.loadtxt(self.path / 'coords.csv', delimiter=',')
        except:
            self.coords = np.loadtxt(self.path / 'initial_coords.csv', delimiter=',')
        with open(self.path / 'conf.yaml', mode='r') as conf_file:
            self.conf = yaml.safe_load(conf_file)

        state_generator = StateGenerator
        self.state_generator = state_generator(G=self.G)
        self.state = self.state_generator.get_initial_state(self.coords)


    @staticmethod
    def new(graph_name: str, embedding_name: str = 'random', state_generator: StateGenerator = StateGenerator, root : Path = temporary_experiments_root):
        name = uuid.uuid4().hex
        path = root / name
        path.mkdir(parents=True)

        if graph_name in organic_graphs:
            G = nx.read_edgelist(graphs_root / graph_name / 'edges.csv', nodetype=int)
        else:
            #graph = Graph.new(graph_name)
            pass
        nx.write_edgelist(G, path / 'edges.csv', data=False)
            

        if embedding_name == 'mercator':
            coords = embedding.mercator(G)
        elif embedding_name == 'random':
            coords = embedding.random(G)
        coords = np.array(list(coords.values()))
        np.savetxt(path / 'initial_coords.csv', coords, delimiter=',')

        conf = {
            'graph_name': graph_name,
            'embedding_name': embedding_name,
            'model': state_generator.__name__,
            'step': 0,
            'time': 0.0,
        }
        with open(path / 'conf.yaml', mode='w') as conf_file:
            yaml.safe_dump(conf, conf_file)

        return GreedyRoutingAnnealer(path)

    @staticmethod
    def load(path: Path):
        G = nx.read_edgelist(path / 'edges.csv', nodetype=int)
        try:
            coords = np.loadtxt(path / 'coords.csv', delimiter=',')
        except:
            coords = np.loadtxt(path / 'initial_coords.csv', delimiter=',')
        with open(path / 'conf.yaml', mode='r') as conf_file:
            conf = yaml.safe_load(conf_file)
        return GreedyRoutingAnnealer(G, coords, StateGenerator, conf)

    def save_experiment(self):
        metrics = self.state_generator.get_metrics()
        
    def embed(self, steps: int, log_freq: int = np.iinfo(int).max):
        log_freq = self.graph.size if log_freq is None else log_freq
        if self.step == 0:
            self.save_experiment()
        start = time.time()
        step = self.conf['step']
        for step in tqdm(range(1, int(steps)+1)):
            new_state = self.state_generator.get_new_state(self.state)
            dE = self.state.grs - new_state.grs
            temp = 1 / (self.step * self.graph.size + 1)
            if dE < 0 or np.random.rand() < np.exp(-dE / temp):
                self.state = new_state
            self.step += 1
            if step % log_freq == 0:
                self.conf['time'] += time.time() - start
                self.conf['step'] = step
                self.save_experiment()
                start = time.time()