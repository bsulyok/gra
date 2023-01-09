from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, truncnorm
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import networkx as nx
import warnings

from .state_generator import StateGenerator

class DegreeDependentSampling(StateGenerator):
    def __init__(self, **args):
        super().__init__(**args)

    def __init__(self, G: nx.Graph):
        super.__init__()
        degree = self.state_generator.adjm.sum(axis=0)
        self.sampling_probability = degree / np.sum(degree)
    
    def select_vertex(self) -> int:
        return np.random.randint(self.N, p=self.sampling_probability)