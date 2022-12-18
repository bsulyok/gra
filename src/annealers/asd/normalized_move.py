import numpy as np
from .base_annealer import BaseAnnealer
    
class NormalizedMove(BaseAnnealer):
    def __init__(self, experiment_path):
        super().__init__(experiment_path)
        self.norm = np.mean
    
    def random_move(self) -> np.ndarray:
        coord = self.coord.copy()
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = self.coord[self.moved_vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        coord[self.moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        coord[self.moved_vertex, 1] = np.random.normal(loc=self.coord[self.moved_vertex, 1], scale=np.pi/4) % (2*np.pi)
        return coord
    
    
def random_move(coord: np.ndarray, moved_vertex: int) -> np.ndarray:
    N = coord.shape[0]
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = coord[moved_vertex, 0], 1
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    coord[moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
    coord[moved_vertex, 1] = np.random.normal(loc=coord[moved_vertex, 1], scale=np.pi/4) % (2*np.pi)
    return coord