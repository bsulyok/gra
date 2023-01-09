import mercator as mercator_embedding # pip install git+https://github.com/networkgeometry/mercator.git@master
from tempfile import TemporaryDirectory
from typing import Dict, Tuple
import networkx as nx
import numpy as np

def mercator(G: nx.Graph) -> Dict[int, Tuple[float]]:
    with TemporaryDirectory() as tempdir:
        nx.write_edgelist(G, f'{tempdir}/graph.edgelist', data=False)
        mercator_embedding.embed(f'{tempdir}/graph.edgelist', clean_mode=True, quiet_mode=True)
        coords = dict(enumerate(map(tuple, np.loadtxt(f'{tempdir}/graph.inf_coord_raw', dtype=float)[:, [2, 1]])))
    return coords

def random(G: nx.Graph) -> Dict[int, Tuple[float]]:
    theta = 2 * np.pi * np.random.rand(len(G))
    degree = np.array(list(map(lambda x: x[1], G.degree())))
    idx = np.argsort(degree + np.random.rand(len(G)))[::-1]
    r = np.empty_like(theta)
    r[idx] = np.log(np.arange(1, len(G)+1))
    return dict(enumerate(map(tuple, np.stack([r, theta]).T)))