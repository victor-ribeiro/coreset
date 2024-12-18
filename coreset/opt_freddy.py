###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

import numpy as np
import heapq
import math
from itertools import batched

from coreset.lazzy_greed import freddy
from coreset.kmeans import kmeans_sampler
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


@timeit
def opt_freddy(dataset, K=1, batch_size=32, max_iter=1000, random_state=None):
    alpha, beta = 0.15, 0.75
    rng = np.random.default_rng(random_state)
    features = dataset.copy().astype(float)
    n, _ = features.shape
    w = np.zeros(n)
    p = np.zeros(n)
    score = np.zeros(n)
    for _ in range(max_iter):
        idx = rng.integers(0, n, int(n * 0.05))
        sample = features[idx]
        util, sset = freddy(
            sample, K=int(n * 0.02), alpha=alpha, beta=beta, return_vals=True
        )
        sset = idx[sset]
        score[sset] += util / len(sset)
        w[sset] += 1
        p[idx] += 1

    _p = 1 - ((p.sum() - p) / p.sum())  # X
    _w = 1 - ((w.sum() - w) / w.sum())  # Y
    _scr = 1 - ((score.max() - score) / score.max())  # Z
    # code = np.zeros((n, 4))
    code = np.zeros((n, 3))
    code[..., 0] = _p
    code[..., 1] = _w
    code[..., 2] = _scr
    # code[..., 0] = (2 * _p) / ((_p**2) + (_w**2) + (_scr**2))
    # code[..., 1] = (2 * _w) / ((_p**2) + (_w**2) + (_scr**2))
    # code[..., 2] = (2 * _scr) / ((_p**2) + (_w**2) + (_scr**2))

    # features = code @ (features.T @ code).T
    # sset = freddy(features, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
    sset = freddy(code, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
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
