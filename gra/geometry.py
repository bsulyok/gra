import numpy as np
import warnings


def euclidean_distance(x1: list[float], x2: list[float]) -> float:
    return np.sqrt(np.sum(np.power(np.subtract(x1, x2), 2), axis=-1))


def cosine_similarity(x1: list[float], x2: list[float]) -> float:
    r1, r2 = np.linalg.norm(x1, axis=-1), np.linalg.norm(x2, axis=-1)
    return np.sum(np.multiply(x1, x2), axis=-1) / r1 / r2


def native_disk_distance_old(x1: list[float], x2: list[float], K: float = -1) -> float:
    zeta = np.sqrt(-K)
    r1 = np.linalg.norm(x1, axis=-1).clip(min=np.finfo(float).resolution)
    r2 = np.linalg.norm(x2, axis=-1).clip(min=np.finfo(float).resolution)
    cosim = np.sum(np.multiply(x1, x2), axis=-1) / r1 / r2
    r1, r2 = r1 * zeta, r2 * zeta
    return np.arccosh(np.clip(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*cosim, a_min=1.0, a_max=None)) / zeta

def native_disk_distance(x1: list[float], x2: list[float], K: float = -1) -> float:
    zeta = np.sqrt(-K)
    r1 = np.linalg.norm(x1, axis=-1).clip(min=np.finfo(float).resolution)
    r2 = np.linalg.norm(x2, axis=-1).clip(min=np.finfo(float).resolution)
    cosim = np.sum(np.multiply(x1, x2), axis=-1) / r1 / r2
    r1, r2 = r1 * zeta, r2 * zeta
    
    #return np.arccosh(np.clip(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*cosim, a_min=1.0, a_max=None)) / zeta

def native_disk_distance(x1: list[float], x2: list[float], K: float = -1) -> float:
    zeta = np.sqrt(-K)
    r1, r2 = np.linalg.norm(x1, axis=-1), np.linalg.norm(x2, axis=-1)
    cosim = np.sum(np.multiply(x1, x2), axis=-1) / r1 / r2
    r1, r2 = r1 * zeta, r2 * zeta
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='invalid value encountered in arccosh')
        distm = np.arccosh(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*cosim) / zeta
    distm[r1 == r2] = 0.0
    zero_idx = np.isnan(distm)
    distm[zero_idx] = np.abs(r1- r2)[zero_idx]
    return distm