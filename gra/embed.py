import numpy as np
from typing import Callable, Iterable, Union, Optional
import networkx as nx
from .state_generators import StateGenerator, RandomSampling
from .geometry import native_disk_distance
from .annealer import Annealer
from .embedding import random as random_embedding
from .score_functions import ScoreFunction, GreedyRoutingSuccessRate
from .temperature import LinearMultiplicativeCooling


def embed(
    G: nx.Graph,
    maxiter: int = None,
    target_function: ScoreFunction = GreedyRoutingSuccessRate,
    state_generator: StateGenerator = RandomSampling,
    temperature: Union[Callable, Iterable] = LinearMultiplicativeCooling(t0=1.0, alpha=1.0),
    exclude_neighbours: bool = False,
    distance_function: Callable = native_disk_distance,
    coords: Optional[dict] = None,
):
    spadjm = nx.adjacency_matrix(G)
    coords_dict = coords or random_embedding(G)
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