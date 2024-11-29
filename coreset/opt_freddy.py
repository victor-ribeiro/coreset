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


def get_labels(sset_idx):
    n = len(sset_idx)
    shape = (n, 2)
    labels = np.zeros(shape)
    labels[:, 1] = 1
    return labels


def info(w_):
    return -((1 - w_) * np.log2(w_)).sum()


def get_weights():
    pass


import matplotlib.pyplot as plt


@timeit
def opt_freddy(
    dataset,
    K=1,
    epochs=15,
    step=10e-3,
    batch_size=32,
    max_iter=500,
    random_state=None,
    tol=10e-3,
    verbose=True,
):
    # _w como vetor de probabilidades e não como matriz nx2
    alpha, beta = 0.15, 0.75
    # alpha, beta = 1, 1
    rng = np.random.default_rng(random_state)
    features = dataset.copy()
    n, m = features.shape
    _w = np.zeros((n, 2))
    # bias = np.random.normal(0, 0.25, (n, m))
    # bias = np.zeros((n, m))
    labels = np.zeros((n, 2))
    for epoch in range(epochs):
        sample = rng.integers(0, n, 200)
        sample = features[sample]
        base_util, _ = freddy(sample, K=30, return_vals=True)
        h = base_util.max()
        e = 0
        _d = 0
        lim = 0
        for _ in range(max_iter):
            idx = rng.integers(0, len(sample), 100)
            ft = sample[idx]
            util, sset = freddy(ft, K=30, alpha=alpha, beta=beta, return_vals=True)
            sset = idx[sset]
            labels[sset] += get_labels(sset)
            h += util.max() - h
            lim = rmse(util, base_util) - e
            # lim = info(util, base_util) - e
            _d += lim / h
            e += lim
        if verbose:
            print(f"[{epoch}] :: {e:.4f}, ({alpha:.4f}, {beta:.4f}, {_d})")

        # _w += (labels.max() - labels) / (labels.max() - labels.min())
        labels += 1
        _w = labels / labels.sum(axis=1).reshape(-1, 1)
        features = _w @ (features.T @ _w).T
        # features += bias

        # bias += step * _d
        alpha += step * _d
        beta += step * _d
        if (abs(lim - e)) < tol:
            break
        # features = _w @ (features.T @ _w).T
    sset = freddy(features, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
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
