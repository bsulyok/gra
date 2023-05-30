from .embedding_score_functions import EmbeddingState, EmbeddingScoreFunction, GreedyRoutingSuccessRate
import numpy as np
from typing import Callable, Iterable, Union, Optional
import csv
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
import networkx as nx
from .embedding_state_generators import EmbeddingStateGenerator, RandomSampling
from tempfile import TemporaryDirectory
from . import embedding
from .geometry import native_disk_distance


__all__ = [
    'ExponentionalMultiplicativeCooling',
    'LogarithmicalMultiplicativeCooling',
    'LinearMultiplicativeCooling',
    'QuadraticMultiplicativeCooling',
    'Annealer',
    'embed',
]


class ExponentionalMultiplicativeCooling(Callable):
    def __init__(self, t0 = 1.0, alpha=0.9):
        self.t0 = t0
        self.alpha = alpha
    def __call__(self, k):
        return self.t0 * self.alpha**k

class LogarithmicalMultiplicativeCooling(Callable):
    def __init__(self, t0 = 1.0, alpha=1.0):
        self.t0 = t0
        self.alpha = alpha
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * np.log(k + 1))

class LinearMultiplicativeCooling(Callable):
    def __init__(self, t0 = 1.0, alpha=1.0):
        self.t0 = t0
        self.alpha = alpha
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * k)

class QuadraticMultiplicativeCooling(Callable):
    def __init__(self, t0 = 1.0, alpha=1.0):
        self.t0 = t0
        self.alpha = alpha
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * k**2)


class Annealer:
    def __init__(self,
            state_generator: Callable,
            target_function: Callable,
            metric_functions: dict[str, Callable] = {},
            path: Optional[Path] = None,
        ):
        self.state_generator = state_generator
        self.target_function = target_function
        self.metric_functions = metric_functions
        self.logs = pd.DataFrame(columns=list(self.metric_functions) + ['step', 'time'])
        self.path = path
        if isinstance(path, str):
            path = Path(path)
        if path is not None:
            path.mkdir(exist_ok=True, parents=True)

    def save_logs(self, state: EmbeddingState, step: int) -> None:
        self.logs.loc[self.logs.shape[0]] = [func(state) for func in self.metric_functions.values()] + [step, time.time()]
        if self.path is not None:
            self.logs.to_csv(self.path / 'logs.csv', index=False)
        
    def embed(self,
        state: EmbeddingState,
        maxiter: int,
        temperature: Union[Callable, Iterable] = LinearMultiplicativeCooling(),
        log_freq: int = np.inf
    ) -> None:
        if isinstance(temperature, Iterable):
            temp_func = lambda step: temperature[step-1]
        else:
            temp_func = temperature
        self.save_logs(state, 0)
        progress_file = None if self.path is None else open(self.path / 'progress.tqdm', 'w')
        for step in tqdm(range(1, int(maxiter)+1), file=progress_file):
            proposed_state = self.state_generator(state)
            self.target_function(proposed_state, state, update=True)
            dE = state.score - proposed_state.score
            T = temp_func(step-1)
            if dE < 0 or np.random.rand() < np.exp(-dE/T):
                state = proposed_state
            if step % log_freq == 0:
                self.save_logs(state, step)
                if state.score == 1.0:
                    break
        return state


def embed(
    G: nx.Graph,
    maxiter: int = None,
    target_function: EmbeddingScoreFunction = GreedyRoutingSuccessRate,
    state_generator: EmbeddingStateGenerator = RandomSampling,
    temperature: Union[Callable, Iterable] = LinearMultiplicativeCooling(t0=1.0, alpha=1.0),
    coords: dict = None,
    exclude_neighbours: bool = False,
    distance_function: Callable = native_disk_distance
):
    spadjm = nx.adjacency_matrix(G)
    coords_dict = coords or embedding.random(G)
    coords = np.array([coords_dict[v] for v in G])
    maxiter = maxiter or 100 * len(G)
    annealer = Annealer(
        state_generator=state_generator(spadjm=spadjm),
        target_function=target_function(
            spadjm=spadjm,
            exclude_neighbours=exclude_neighbours,
            distance_function=distance_function
        )
    )
    initial_state = annealer.target_function.initial_state(coords)
    state = annealer.embed(state=initial_state, maxiter=maxiter, temperature=temperature)
    coords_emb = dict(zip(G, state.coords.tolist()))
    return coords_emb

    

def embed2(
    G: nx.Graph,
    maxiter: int = None,
    target_function: EmbeddingScoreFunction = GreedyRoutingSuccessRate,
    state_generator: EmbeddingStateGenerator = RandomSampling,
    temperature: Union[Callable, Iterable] = None,
    coords: dict = None,
    exclude_neighbours: bool = False,
    distance_function: Callable = native_disk_distance
):  
    spadjm = nx.adjacency_matrix(G)
    coords_dict = coords or embedding.random(G)
    coords = np.array([coords_dict[v] for v in G])
    maxiter = maxiter or 100 * len(G)
    if temperature is None:
        temperature = lambda step: 1/(step)
    with TemporaryDirectory() as experiment_path:
        annealer = Annealer(
            state_generator=state_generator(spadjm=spadjm),
            target_function=target_function(
                spadjm=spadjm,
                exclude_neighbours=exclude_neighbours,
                distance_function=distance_function
            ),
            experiment_path=Path(experiment_path),
        )
        initial_state = annealer.target_function.initial_state(coords)
        state = annealer.embed(
            state=initial_state,
            maxiter=maxiter,
            temperature=temperature
        )
    coords_emb = dict(zip(G, state.coords.tolist()))
    return coords_emb