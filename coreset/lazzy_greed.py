###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################
import heapq
import math
import numpy as np
from numba import jit
from typing import Any
from itertools import batched

from coreset.metrics import METRICS
from coreset.utils import timeit
from coreset.dataset.dataset import Dataset

REDUCE = {"mean": np.mean, "sum": np.sum}


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

    def push(self, idx, score):
        item = (idx, score)
        self.append(item)


def base_inc(alpha=1):
    # mudar isso aqui
    return math.log(1 + alpha)


def utility_score(e, sset, /, alpha=1, reduce="mean"):
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1)
    return norm * math.log(1 + REDUCE[reduce](argmax) * f_norm)


@timeit
def lazy_greed(
    dataset,
    base_inc=base_inc,
    alpha=1,
    metric="similarity",
    K=1,
    reduce_fn="sum",
    batch_size=32,
):
    # basic config
    base_inc = base_inc(alpha)
    argmax = np.zeros(len(dataset))
    idx = np.arange(len(dataset))
    score = 0
    q = Queue()
    sset = []
    vals = []

    for D, V in zip(
        METRICS[metric](dataset, batch_size=batch_size),
        batched(idx, batch_size),
    ):
        size = len(D)
        [q.push(base_inc, idx) for idx in zip(V, range(size))]
        while len(sset) < K and q:
            _, idx_s = q.head
            s = D[idx_s[1]]
            score_s = utility_score(s, argmax, alpha=alpha, reduce=reduce_fn)
            inc = score_s - score
            if inc < 0:
                continue
                # break
            if not q:
                break
            score_t, idx_t = q.head
            if inc > score_t:
                argmax = np.maximum(argmax, s)
                score = utility_score(s, argmax, alpha=alpha, reduce=reduce_fn)
                sset.append(idx_s[0])
                vals.append(score)
                q.push(score_t, idx_t)
            q.push(inc, idx_s)
        else:
            argmax = np.maximum(argmax, s)
            score = utility_score(s, argmax, alpha)
            sset.append(idx_s[0])
            vals.append(score)

    return sset, vals
