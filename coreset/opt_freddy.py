###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

import numpy as np
import heapq
import math
from itertools import batched

from coreset.lazzy_greed import freddy
from coreset.metrics import METRICS
from coreset.utils import timeit


def rmse(y, y_hat):
    y, y_hat = np.array(y), np.array(y_hat)
    error = (y - y_hat) ** 2
    error = error.sum() / len(y)
    return error**0.5


def get_labels(sset_idx, n):
    shape = (n, 2)
    idx = np.zeros(n, dtype=bool)
    idx[sset_idx] = True
    labels = np.zeros(shape)
    labels[idx, 1] = 1
    labels[~idx, 0] = 1
    return labels


def get_weights():
    pass


import matplotlib.pyplot as plt


@timeit
def opt_freddy(
    dataset, K=1, epochs=15, step=10e-4, batch_size=32, max_iter=100, random_state=None
):
    alpha, beta = 0.25, 0.75
    n = len(dataset)
    rng = np.random.default_rng(random_state)
    # _w = np.zeros((n, 2))
    for epoch in range(epochs):
        # sample = rng.integers(0, n, n // 5)
        sample = rng.integers(0, n, 500)
        sample = dataset[sample]
        base_util, _ = freddy(
            sample,
            K=30,
            return_vals=True,
        )
        h = base_util.max()
        e = 0
        _d = 0
        lim = 0
        for _ in range(max_iter):
            idx = rng.integers(0, len(sample), 100)
            ft = sample[idx]
            util, _ = freddy(ft, K=30, alpha=alpha, beta=beta, return_vals=True)
            h += -util.max()
            lim = rmse(util, base_util) - e
            _d += lim / h
            e += lim
        print(f"[{epoch}] :: {e:.4f}, ({alpha:.4f}, {beta:.4f}, {_d})")
        if abs(lim) < 10e-3:
            break
        alpha += step * _d
        beta += step * _d

        # if len(loss) > 2 and abs(loss[-1] - loss[-2]) < 10e-3:
        #     break
    sset = freddy(dataset, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
    np.random.shuffle(sset)
    return sset


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # n_samples = 10_000
    n_samples = 10_000
    ft, tgt = make_blobs(
        n_samples=n_samples,
        n_features=2,
        cluster_std=0.5,
        center_box=(-5, 5),
        random_state=42,
    )
    # import matplotlib.pyplot as plt

    # plt.scatter(*ft.T)
    # plt.show()
    out = opt_freddy(ft, K=int(len(ft) * 0.1), epochs=500, step=1)
