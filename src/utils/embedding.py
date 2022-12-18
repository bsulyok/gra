from pathlib import Path
from .graph import Graph
import numpy as np
import mercator
from tempfile import TemporaryDirectory


def mercator_embedding(G: Graph) -> np.ndarray:
    with TemporaryDirectory() as tempdir:
        np.savetxt(f'{tempdir}/graph.edgelist', G.edges, delimiter=' ', fmt='%i')
        mercator.embed(f'{tempdir}/graph.edgelist', clean_mode=True, quiet_mode=True)
        coords = np.loadtxt(f'{tempdir}/graph.inf_coord')[:, 3:1:-1]
    return coords


def random_embedding(G: Graph) -> np.ndarray:
    N = G.size
    coords = np.empty((N, 2), dtype=float)
    radial_coord = np.log(np.arange(1, N+1)) + np.log(N)
    degree_order = np.argsort(np.sum(G.adjm, axis=0) + np.random.rand(N))[::-1]
    coords[degree_order, 0] = radial_coord
    coords[:, 1] = np.random.rand(N) * 2 * np.pi
    return coords

def stupid_random_embedding(G: Graph) -> np.ndarray:
    N = G.size
    coords = np.empty((N, 2), dtype=float)
    radial_coord = np.log(np.arange(1, N+1)) + np.log(N)
    np.random.sort(radial_coord)
    coords[:, 0] = radial_coord
    coords[:, 1] = np.random.rand(N) * 2 * np.pi
    return coords