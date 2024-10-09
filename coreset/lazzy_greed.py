###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################
from numba import jit
import heapq
import math
import numpy as np
from itertools import batched

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


def utility_score(e, sset, /, alpha=1, reduce="mean"):
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1)
    # return norm * math.log(1 + REDUCE[reduce](argmax) * f_norm)
    return (
        norm * math.log(1 + REDUCE[reduce](argmax) * f_norm)
        + (0.5 * argmax.sum())
        + (0.5 * np.linalg.norm(argmax))
    )


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
    ## tentativa de ajuste multiplicando a função de utilidade pelo distânca entre o novo elemento
    ## e as maiores distâncias
    # basic config
    base_inc = base_inc(alpha)
    idx = np.arange(len(dataset))
    argmax = np.zeros(batch_size)
    q = Queue()
    sset = []
    vals = []
    alphas = []
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        if len(ds) < batch_size:
            break
        D = METRICS[metric](ds, batch_size=batch_size)
        # D += 0.5 * (D**2).sum()
        # D += 0.5 * (np.linalg.norm(D, axis=1)).sum()
        size = len(D)
        [q.push(base_inc, i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = utility_score(s, argmax, alpha=alpha, reduce=reduce_fn)
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                argmax = np.maximum(argmax, s)
                # score = utility_score(s, argmax, alpha=alpha, reduce=reduce_fn) * (
                #     np.linalg.norm(argmax - s)
                # )
                score = utility_score(s, argmax, alpha=alpha, reduce=reduce_fn)
                # score += 0.5 * (argmax**2).sum() #-> L2
                # score += 0.5 * np.linalg.norm(argmax, ord="fro")
                sset.append(idx_s[0])
                vals.append(score)
                q.push(inc, idx_t)
            else:
                q.push(inc, idx_s)
    return np.array(sset)
