###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################
from numba import jit
import heapq
import math
import numpy as np
from itertools import batched
import random

from coreset.metrics import METRICS
from coreset.utils import timeit

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
    return math.log(1 + alpha)


def utility_score(e, sset, /, acc=0, alpha=1, reduce="mean"):
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1 + acc)
    util = norm * math.log(1 + (argmax.sum() + acc) * f_norm)
    return util + math.log(1 + (sset**2).sum())


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
    idx = np.arange(len(dataset))
    q = Queue()
    sset = []
    vals = []
    argmax = 0
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        D = METRICS[metric](ds, batch_size=batch_size)
        size = len(D)
        # localmax = np.median(D, axis=0)
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()
        _ = [q.push(base_inc, i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = utility_score(
                s, localmax, acc=argmax, alpha=alpha, reduce=reduce_fn
            )
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                score = utility_score(
                    s, localmax, acc=argmax, alpha=alpha, reduce=reduce_fn
                )
                localmax = np.maximum(localmax, s)
                sset.append(idx_s[0])
                vals.append(score)
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)
    sset = np.array(sset)
    return sset


def lazy_greed_class(
    features,
    targets,
    base_inc=base_inc,
    alpha=1,
    metric="similarity",
    K=1,
    reduce_fn="sum",
    batch_size=32,
):
    classes = np.unique(targets)
    n_class = len(classes)
    idx = np.arange(len(features))
    idx = [np.where(targets == c) for c in classes]
    sset = [
        lazy_greed(
            features[i],
            base_inc,
            alpha,
            metric,
            int(K / n_class),
            reduce_fn,
            batch_size,
        )
        for i in idx
    ]
    sset = np.hstack(sset)
    np.random.shuffle(sset)
    return sset
