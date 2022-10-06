import numpy as np
from .base_annealer import BaseAnnealer

class CloggingDependent(BaseAnnealer):
    def __init__(self, graph_name, coord_name):
        super().__init__(graph_name, coord_name)
        
    def select_for_move(self) -> np.ndarray:
        failed_packets = self.gd[self.gd != np.arange(self.N)]
        bincount = np.bincount(failed_packets, minlength=self.N)
        bincount += 1
        return np.random.choice(self.N, p=bincount/bincount.sum())