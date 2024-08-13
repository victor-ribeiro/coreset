###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

import heapq
import numpy as np
from functools import lru_cache, reduce
from coreset.utils.metrics import *


class Queue(list):
    def __init__(self, *iterable):
        super().__init__(*iterable)
        heapq._heapify_max(self)

    def append(self, item: "Any"):
        super().append(item)
        heapq._siftdown_max(self, 0, len(self) - 1)

    def pop(self, index=-1):
        el = super().pop(index)
        if not self:
            return el
        val, self[0] = self[0], el
        heapq._siftup_max(self, 0)
        return val

    @property
    def head(self):
        return self.pop()

    def push(self, item):
        self.append(item)


@lru_cache
def L_s0(alpha=1):
    return np.log(1 + alpha)


def sset_norm(diff_mtx):
    max_nom = map(lambda x: x.max(axis=1), diff_mtx)


def utility_score(e, argmax, base_loc):
    norm = 1 / base_loc
    F_s = norm * np.log(1 + np.maximum(e, argmax))


if __name__ == "__main__":
    import random

    coisa = [(np.random.normal(0, 1), i) for i in range(5)]
    random.shuffle(coisa)
    print(f"{coisa=}")
    fila = Queue()
    while coisa:
        item = coisa.pop()
        fila.push(item)

        # print(f"{fila=}")
    while fila:
        val = fila.head
        print(val)
