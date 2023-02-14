import mercator as mercator_embedding # pip install git+https://github.com/networkgeometry/mercator.git@master
from tempfile import TemporaryDirectory
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def mercator(G: nx.Graph, inplace=False) -> Dict[int, Tuple[float]]:
    with TemporaryDirectory() as tempdir:
        nx.write_edgelist(G, f'{tempdir}/graph.edgelist', data=False)
        mercator_embedding.embed(f'{tempdir}/graph.edgelist', clean_mode=True, quiet_mode=True)
        inf_coord = pd.read_csv(f'{tempdir}/graph.inf_coord', delim_whitespace=True, comment='#', names=['vertex', 'kappa', 'theta', 'r'])
        inf_coord['r'] *= 2*np.log(len(G)) / inf_coord['r'].max()
        coords = dict(zip(inf_coord['vertex'], map(list, inf_coord[['r', 'theta']].values)))
    if inplace:
        nx.set_node_attributes(G, coords, name='coords')
    else:
        return coords


def random(G: nx.Graph, inplace: bool = False) -> Dict[int, Tuple[float]]:
    theta = 2 * np.pi * np.random.rand(len(G))
    degree = np.array(list(map(lambda x: x[1], G.degree())))
    idx = np.argsort(degree + np.random.rand(len(G)))[::-1]
    r = np.empty_like(theta)
    r[idx] = np.log(np.arange(1, len(G)+1)) + np.log(len(G))
    coords = dict(zip(G.nodes, map(tuple, np.stack([r, theta]).T)))
    if inplace:
        nx.set_node_attributes(G, coords, name='coords')
    else:
        return coords

def preset(G: nx.Graph, inplace: bool = False) -> Dict[int, Tuple[float]]:
    coords = nx.get_node_attributes(G, name='coords')
    return coords