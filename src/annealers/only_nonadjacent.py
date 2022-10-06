import numpy as np
from .base_model import BaseAnnealer

class OnlyNonadjacent(BaseAnnealer):
    def __init__(self, adjacency_matrix, coord=None):
        super().__init__(adjacency_matrix, coord)
    
    def greedy_routing_score(self, gd: np.ndarray) -> float:
        successful = np.sum(gd == np.arange(self.N))
        return (successful - self.N - self.K) / (self.N * (self.N-1) - self.K)
    
    def energy_change(self, gd) -> float:
        successful = np.sum(gd == np.arange(self.N))
        grs = (successful - self.N - self.K) / (self.N * (self.N-1) - self.K)
        
        prev_successful = np.sum(self.gd == np.arange(self.N))
        prev_grs = (successful - self.N - self.K) / (self.N * (self.N-1) - self.K)
        
        return (prev_grs - grs) * (self.N - self.K)