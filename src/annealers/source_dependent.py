import numpy as np
from .base_annealer import BaseAnnealer
    
class SourceDependent(BaseAnnealer):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_for_move(self) -> int:
        unsuccessful = np.sum(self.gd != np.arange(self.N), axis=1)
        unsuccessful += 1
        return np.random.choice(self.N, p=unsuccessful / np.sum(unsuccessful))