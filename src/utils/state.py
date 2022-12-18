from .graph import Graph
from .embedding import mercator_embedding, random_embedding
from typing import Callable
from pathlib import Path
import numpy as np
import uuid
import csv
import time
import yaml
from dataclasses import dataclass
from .functions import initial_distance, initial_greedy_closest_neighbour, initial_greedy_destination, greedy_routing_score


@dataclass
class State:
    name: str
    embedding_name: str
    coords: np.ndarray
    distance: np.ndarray
    gcn: np.ndarray
    gd: np.ndarray
    grs: float
    step: int
    time: float  
    
    @staticmethod
    def new(graph: Graph, embedding_name: str):# -> GRAState:
        name = uuid.uuid4().hex
        if embedding_name == 'mercator':
            start = time.time()
            coords = mercator_embedding(graph)
            print(time.time() - start)
        elif embedding_name == 'random':
            coords = random_embedding(graph)
        else:
            raise 'this embedding method is not implemented'
        distance = initial_distance(coords)
        gcn = initial_greedy_closest_neighbour(distance, graph.adjl)
        gd = initial_greedy_destination(gcn)
        grs = greedy_routing_score(gd)
        step = 0
        time = 0.0
        return State(name, embedding_name, coords, distance, gcn, gd, grs, step, time)
    
    @staticmethod
    def load(path: Path, graph: Graph):# -> GRAState:
        with open(path / 'conf.yaml', mode='r') as conf_file:
            conf = yaml.safe_load(conf_file)
        name = conf['name']
        embedding_name = conf['embedding_name']
        step = conf['step']
        time = conf['time']
        try:
            coords = np.loadtxt(path / 'coords.csv', delimiter=',')
        except:
            coords = np.loadtxt(path / 'initial_coords.csv', delimiter=',')
        distance = initial_distance(graph.coords)
        gcn = initial_greedy_closest_neighbour(distance, graph.adjl)
        gd = initial_greedy_destination(gcn)
        grs = greedy_routing_score(gd)
        return State(name, embedding_name, coords, distance, gcn, gd, grs, step, time)
    
    def write(self, path: Path, metrics: list) -> None:
        # write logs
        data_dict = {
            'step': int(self.step),
            'time': float(self.time),
            'grs': float(self.grs)
        }
        data_dict.update(metrics)
        logs_path = path / 'logs.csv'
        if not logs_path.exists():
            with open(logs_path, 'a') as logs_file:
                csv.writer(logs_file, delimiter=',').writerow(data_dict.keys())
        with open(logs_path, 'a') as logs_file:
            csv.writer(logs_file, delimiter=',').writerow(data_dict.values())
        # write conf
        conf_path = path / 'conf.yaml'
        if conf_path.exists():
            with open(conf_path, 'r') as conf_file:
                conf = yaml.safe_load(conf_file)
                conf.update(data_dict)
        else:
            conf = {}
        conf.update({'name': self.name, 'embedding_name': self.embedding_name})
        conf.update(data_dict)
        with open(conf_path, 'w') as conf_file:
            yaml.safe_dump(conf, conf_file)