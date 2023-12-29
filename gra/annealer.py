from .score_functions import State, ScoreFunction, GreedyRoutingSuccessRate
import numpy as np
from typing import Callable, Iterable, Union, Optional
import csv
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
import networkx as nx
from .state_generators import StateGenerator, RandomSampling
from tempfile import TemporaryDirectory
from . import embedding
from .geometry import native_disk_distance


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
        self.logs = pd.DataFrame(columns=['step', 'time'] + list(self.metric_functions))
        self.path = path
        if isinstance(path, str):
            path = Path(path)
        if path is not None:
            path.mkdir(exist_ok=True, parents=True)

    def save_logs(self, state: State, step: int) -> None:
        self.logs.loc[self.logs.shape[0]] = [step, time.time()] + [func(state) for func in self.metric_functions.values()]
        if self.path is not None:
            self.logs.to_csv(self.path / 'logs.csv', index=False)
        
    def embed(self,
        state: State,
        maxiter: int,
        temperature: Union[Callable, Iterable],
        log_freq: int = np.inf,
        stopping_criteria: Optional[callable] = None
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
                if stopping_criteria is not None and stopping_criteria(state):
                    break
        return state