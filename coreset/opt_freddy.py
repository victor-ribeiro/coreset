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
    alpha, beta = 0.5, 0.75
    rng = np.random.default_rng(random_state)
    features = dataset.copy().astype(float)
    n, _ = features.shape
    w = np.zeros(n)
    p = np.zeros(n)
    score = np.zeros(n)

    for _ in range(max_iter):
        # idx = rng.integers(0, n, 3000)
        idx = rng.integers(0, n, int(n * 0.05))
        sample = features[idx]
        util, sset = freddy(
            sample, K=int(n * 0.01), alpha=alpha, beta=beta, return_vals=True
        )
        sset = idx[sset]
        score[sset] += util / len(sset)
        w[sset] += 1
        p[idx] += 1

    _p = 1 - (p / p.sum())  # X
    _w = 1 - (w / w.sum())  # Y
    _scr = 1 - (score / score.max())  # Z
    # code = np.zeros((n, 4))
    code = np.zeros((n, 3))
    code[..., 0] = (2 * _p) / ((_p**2) + (_w**2) + (_scr**2))
    code[..., 1] = (2 * _w) / ((_p**2) + (_w**2) + (_scr**2))
    code[..., 2] = (2 * _scr) / ((_p**2) + (_w**2) + (_scr**2))

    features = code @ (features.T @ code).T
    # sset = freddy(features, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
    sset = freddy(code, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
    return sset
    # step = 10e-4
    # rng = np.random.default_rng(random_state)
    # features = dataset.copy()
    # n, m = features.shape
    # _w = np.zeros((n, 2))
    # labels = np.zeros((n, 2))
    # for epoch in range(10):
    #     sample = rng.integers(0, n, 2000)
    #     sample = features[sample]
    #     base_util, _ = freddy(sample, K=30, return_vals=True)
    #     h = base_util.max()
    #     e = 10e-8
    #     _d = 10e-8
    #     lim = 0
    #     for _ in range(max_iter):
    #         idx = rng.integers(0, len(sample), 100)
    #         ft = sample[idx]
    #         util, sset = freddy(ft, K=30, alpha=alpha, beta=beta, return_vals=True)
    #         sset = idx[sset]
    #         labels[sset] += get_labels(sset)
    #         h += (util.max() - base_util.max()) / base_util.max()
    #         lim = rmse(util, base_util)
    #         # lim = info(util, base_util) - e
    #         _d += lim / h
    #         e += lim
    #     # if e < 10e-4:
    #     #     break
    #     print(f"[{epoch}] :: {e:.4f}, ({alpha:.4f}, {beta:.4f}, {_d})")
    #     labels += 1
    #     _w = labels / labels.sum(axis=1).reshape(-1, 1)
    #     features -= _w @ (features.T @ _w).T * step
    #     alpha -= step  # * _d
    #     beta -= step  # * _d
    #     # alpha -= step * _d
    #     # beta -= step * _d

    # return sset


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
