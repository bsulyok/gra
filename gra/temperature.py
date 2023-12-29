from typing import Callable
import numpy as np


class ExponentionalMultiplicativeCooling(Callable):
    def __init__(self, t0: float = 1.0, alpha: float = 0.9, offset: float = 0.0):
        self.t0 = t0
        self.alpha = alpha
        self.offset = offset
    def __call__(self, k):
        return self.t0 * self.alpha**(k+self.offset)

class LogarithmicalMultiplicativeCooling(Callable):
    def __init__(self, t0: float = 1.0, alpha: float = 1.0, offset: float = 0.0):
        self.t0 = t0
        self.alpha = alpha
        self.offset = offset
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * np.log(k + self.offset + 1))

class LinearMultiplicativeCooling(Callable):
    def __init__(self, t0: float = 1.0, alpha: float = 1.0, offset: float = 0.0):
        self.t0 = t0
        self.alpha = alpha
        self.offset = offset
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * (k + self.offset) )

class QuadraticMultiplicativeCooling(Callable):
    def __init__(self, t0: float = 1.0, alpha: float = 1.0, offset: float = 0.0):
        self.t0 = t0
        self.alpha = alpha
        self.offset = offset
    def __call__(self, k):
        return self.t0 / (1 + self.alpha * (k+self.offset)**2)
