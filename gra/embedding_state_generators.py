import numpy as np
from scipy.stats import truncnorm
from dataclasses import dataclass
from scipy.sparse import spmatrix
from typing import Callable
from .geometry import native_disk_distance


@dataclass
class EmbeddingState:
    coords: np.ndarray
    distance: np.ndarray = None
    score: float = None


__all__ = [
    'EmbeddingState',
    'EmbeddingStateGenerator',
    'RandomSampling',
    'DegreeDependentSampling',
    'SourceDependentSampling',
    'TargetDependentSampling'
]


class EmbeddingStateGenerator:
    def __init__(self, spadjm: spmatrix, distance_function: Callable = native_disk_distance):
        self.spadjm = spadjm
        self.distance_function = distance_function
        self.N = self.spadjm.shape[0]

    def select_vertices(self, state):
        raise NotImplementedError
    
    def move_vertices(self, coords, vertices):
        r = np.linalg.norm(coords[vertices], axis=1)
        theta = np.arctan2(*coords[vertices].T)
        clip_min, clip_max = 1, 2*np.log(self.N)
        loc, scale = r, 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        r = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        theta = np.random.normal(loc=theta, scale=np.pi/4) % (2*np.pi)
        proposed_coords = coords.copy()
        proposed_coords[vertices, 0] = r * np.sin(theta)
        proposed_coords[vertices, 1] = r * np.cos(theta)
        return proposed_coords

    def __call__(self, state):
        moved_vertices = self.select_vertices(state)
        new_coords = self.move_vertices(state.coords, moved_vertices)
        new_state = EmbeddingState(coords=new_coords)
        return new_state
    

class RandomSampling(EmbeddingStateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
        
    def select_vertices(self, state: EmbeddingState):
        moved_vertices = np.random.randint(self.N, size=1)
        return moved_vertices
    

class DegreeDependentSampling(EmbeddingStateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
        degree = self.spadjm.sum(axis=0)
        self.sampling_probability = degree / np.sum(degree)
    
    def select_vertices(self, state: EmbeddingState) -> int:
        return np.random.choice(self.N, p=self.sampling_probability, size=1)


class SourceDependentSampling(EmbeddingStateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_vertices(self, state: EmbeddingState) -> int:
        unsuccessful_paths = np.sum(state.destination != self.arange, axis=1)
        unsuccessful_paths += 1
        return np.random.choice(self.N, p=unsuccessful_paths / np.sum(unsuccessful_paths), size=1)


class TargetDependentSampling(EmbeddingStateGenerator):
    def __init__(self, **args):
        super().__init__(**args)
    
    def select_vertices(self, state: EmbeddingState) -> int:
        unsuccessful_paths = np.sum(state.destination != self.arange, axis=0)
        unsuccessful_paths += 1
        return np.random.choice(self.N, p=unsuccessful_paths / np.sum(unsuccessful_paths), size=1)


__all__ = [
    'RandomSampling',
    'DegreeDependentSampling',
    'SourceDependentSampling',
    'TargetDependentSampling'
]