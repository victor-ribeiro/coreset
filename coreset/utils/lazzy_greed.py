###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

from typing import Callable
import heapq
import numpy as np
from collections import UserList

from coreset.utils.dataset import Dataset
from coreset.utils.metrics import pdist


class Coreset(Dataset):

    def __init__(self, name="") -> None:
        super().__init__(None, name)

    def __add__(self, elemen):
        pass

    def __radd__(self, elemen):
        return self.__add__(elemen)


def _push(heap, e):
    heap.append(e)
    heapq._siftdown_max(heap, e, len(heap))


def queue(V, util_func=None):
    order = []
    heapq.heapify(order)
    [_push(order, i) for i in range(len(V))]
    yield from order


def lazzygreed(
    dataset, size: int | float, tol: float, utility_function: Callable, sample=1
):
    if not bool(size) ^ bool(tol):
        raise ValueError
    for D in pdist(dataset):
        pass


if __name__ == "__main__":
    import numpy as np

    new = []

    q = np.random.normal(0, 100, 10).astype(int).tolist()
    q = queue(q)
    for coisa in q:
        print(coisa)

    # print(q)
