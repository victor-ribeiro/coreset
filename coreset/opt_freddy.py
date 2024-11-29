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
    rng = np.random.default_rng(random_state)
    features = dataset.copy()
    n, m = features.shape
    _w = np.zeros((n, 1))
    labels = np.zeros((n, 2))

    for _ in range(max_iter):
        sample = rng.integers(0, n, 200)
        sample = features[sample]
        idx = rng.integers(0, len(sample), 100)
        ft = sample[idx]
        util, sset = freddy(ft, K=30, alpha=alpha, beta=beta, return_vals=True)
        sset = idx[sset]
        labels[sset] += get_labels(sset)
        features = _w @ (features.T @ _w).T
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
