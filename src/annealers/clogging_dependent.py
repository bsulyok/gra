import numpy as np
from .base_annealer import BaseAnnealer

class CloggingDependent(BaseAnnealer):
    def __init__(self, **args):
        super().__init__(**args)
        self.considered = np.zeros(self.N)
        
    def select_for_move(self) -> np.ndarray:
        failed_packets = self.gd[self.gd != np.arange(self.N)]
        bincount = np.bincount(failed_packets, minlength=self.N)
        bincount = np.maximum(bincount - self.considered, 0) + 1
        return np.random.choice(self.N, p=bincount/bincount.sum())
    
    
    def update(self):
        dE = self.energy_change()
        if dE < 0 or np.random.rand() < np.exp(-dE / self.temp):
            self.coord = self.new_coord
            self.distance = self.new_distance
            self.gcn = self.new_gcn
            self.gd = self.new_gd
            self.grs = self.new_grs
            self.considered = np.zeros(self.N)
        else:
            self.considered[self.moved_vertex] += 1