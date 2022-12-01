import numpy as np
from .base_annealer import BaseAnnealer

class DegreeDependent(BaseAnnealer):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_for_move(self) -> int:
        degree = np.sum(self.adjacency_matrix, axis=0)
        p = degree / np.sum(degree)
        return np.random.choice(self.N, p=p)