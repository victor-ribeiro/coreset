###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

import numpy as np
import heapq
import math
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


def utility_score(e, sset, /, acc=0, alpha=0.1, beta=1.1, gamma=2):
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1 + acc)
    util = norm * math.log(1 + (argmax.sum() + acc) * f_norm)
    return util + (math.log(1 + ((sset.sum() + acc) ** gamma)) * beta)
    # util = norm * math.log(1 + (argmax.sum()) * f_norm)
    # return util + (math.log(1 + ((sset.sum()) ** gamma)) * beta)


@timeit
def freddy(
    dataset,
    base_inc=base_inc,
    # alpha=0.15,
    alpha=1,
    gamma=0.15,
    metric="similarity",
    K=1,
    batch_size=32,
    # beta=0.75,
    beta=1,
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
        inc = 0
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = utility_score(
                s, localmax, acc=argmax, alpha=alpha, beta=beta, gamma=gamma
            )
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                score = utility_score(
                    s, localmax, acc=argmax, alpha=alpha, beta=beta, gamma=gamma
                )
                localmax = np.maximum(localmax, s)
                sset.append(idx_s[0])
                vals.append(score)
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)
        _w = (vals[-1] - vals[0]) / (base_inc - vals[-1])
        alpha -= _w
        beta += _w
    # sset = np.array(sset)
    np.random.shuffle(sset)
    import matplotlib.pyplot as plt

    # plt.plot(vals)
    # plt.show()
    # exit()
    return sset


def lazy_greed_class(
    features,
    targets,
    base_inc=base_inc,
    alpha=1,
    metric="similarity",
    K=1,
    batch_size=32,
    beta=1,
):

    classes, w = np.unique(targets, return_counts=True)
    n_class = len(classes)
    idx = np.arange(len(features))
    sset = []
    k = int(K // n_class)
    for c, w_ in zip(classes, w):
        idx_ = idx[targets.astype(int) == c]
        if k > len(idx_):
            sset.append(idx_)
            continue
        f_ = features[idx_]
        s_ = freddy(f_, base_inc, alpha, metric, k, batch_size, beta=beta)
        sset.append(idx_[s_])

    sset = [idx[i] for i in sset]
    sset = np.hstack(sset)
    np.random.shuffle(sset)
    return sset
