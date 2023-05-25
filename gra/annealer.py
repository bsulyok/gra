from .embedding_score_functions import EmbeddingState, EmbeddingScoreFunction, GreedyRoutingSuccessRate
import numpy as np
from typing import Callable, Iterable, Union
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


class Annealer:
    def __init__(self,
            state_generator: Callable,
            target_function: Callable,
            experiment_path: Path,
            metric_functions: dict[str, Callable] = {},
        ):
        self.state_generator = state_generator
        self.target_function = target_function
        self.experiment_path = experiment_path
        self.metric_functions = metric_functions

    def write_logs(self, state: EmbeddingState, step: int) -> None:
        save_path = self.experiment_path / 'logs.csv'
        if not save_path.exists():
            csv.writer(open(save_path, 'w')).writerow(list(self.metric_functions) + ['step', 'time'])
        csv.writer(open(save_path, 'a')).writerow([func(state) for func in self.metric_functions] + [step, time.time()])

    @property
    def logs(self) -> pd.DataFrame:
        return pd.read_csv(self.experiment_path / 'logs.csv')

    def embed(self,
        state: EmbeddingState,
        maxiter: int,
        temperature: Union[Callable, Iterable],
        log_freq: int = np.inf
    ) -> None:
        if isinstance(temperature, Iterable):
            temp_func = lambda step: temperature[step-1]
        else:
            temp_func = temperature
        self.write_logs(state, 0)
        for step in tqdm(range(1, int(maxiter)+1), file=open(self.experiment_path / 'progress.tqdm', 'w')):
            proposed_state = self.state_generator(state)
            self.target_function(proposed_state, state, update=True)
            dE = state.score - proposed_state.score
            T = temp_func(step)
            if dE < 0 or np.random.rand() < np.exp(-dE/T):
                state = proposed_state
            if step % log_freq == 0:
                self.write_logs(state, step)
                if state.score == 1.0:
                    break
        return state
    

def embed(
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
    temperature = temperature or 1/(np.arange(1, maxiter+1))
    with TemporaryDirectory() as experiment_path:
        annealer = Annealer(
            state_generator=state_generator(spadjm=spadjm),
            target_function=target_function(spadjm, exclude_neighbours=exclude_neighbours, distance_function=native_disk_distance),
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