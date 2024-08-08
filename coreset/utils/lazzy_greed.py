#############################################
### |S| <= 1 + log max F( e | [] ) | S* | ###
#############################################

from typing import Callable
import numpy as np
import heapq

from coreset.utils.dataset import Dataset
from coreset.utils.metrics import pdist


def facility_location(subset: np.array):
    return subset.max(axis=1).sum()


def lazzygreed(
    dataset, size: int | float, tol: float, utility_function: Callable, sample=1
):
    if not bool(size) ^ bool(tol):
        raise ValueError


def submodular(dataset):
    pass


class Coreset(Dataset):

    def __init__(self, name="") -> None:
        super().__init__(None, name)

    def __add__(self, elemen):
        pass

    def __radd__(self, elemen):
        return self.__add__(elemen)
