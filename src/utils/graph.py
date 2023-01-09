from typing import Callable
from ..pso import PSO
import networkx as nx
from pathlib import Path
import numpy as np
import yaml
from dataclasses import dataclass
import warnings


graphs_root = Path('/home/bsulyok/documents/projects/gra/graphs')


@dataclass
class Graph:
    name: str
    size: int
    adjm: np.ndarray
    adjl: np.ndarray
    edges: np.ndarray

    @staticmethod
    def new(name):# -> Graph:
        if name.startswith('pso_'):
            size = int(name[4:])
            G = PSO(size, 2)
            assert nx.is_connected(G)
            edges = np.array(G.edges)
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                adjm = nx.adjacency_matrix(G).toarray().astype(bool)
        adjl = np.array([np.where(neigh)[0] for neigh in adjm], dtype=object)
        return Graph(name, size, adjm, adjl, edges)

    @staticmethod
    def load(path: Path):# -> Graph:
        with open(path / 'conf.yaml', mode='r') as conf_file:
            conf = yaml.safe_load(conf_file)
        name = conf['graph_name']
        size = conf['size']
        edges = np.loadtxt(path / 'edges.csv', delimiter=',', dtype=int)
        adjm = np.zeros((size, size), dtype=bool)
        adjm[edges[:, 0], edges[:, 1]] = adjm[edges[:, 1], edges[:, 0]] = True
        adjl = np.array([np.where(row)[0] for row in adjm], dtype=object)
        return Graph(name, size, adjm, adjl, edges)