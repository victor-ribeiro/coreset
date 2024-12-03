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


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


@timeit
def opt_freddy(dataset, K=1, batch_size=32, max_iter=1000, random_state=42, tol=10e-3):
    # _w como vetor de probabilidades e não como matriz nx2
    alpha, beta = 1, 0.75
    rng = np.random.default_rng(random_state)
    features = dataset.copy()
    n, _ = features.shape
    loss = []

    for __ in range(10):
        e = 0
        h = 10e-6
        d = 10e-6
        prev = None
        w = np.zeros((n, 1))
        for _ in range(max_iter):
            idx = rng.integers(0, n, 300)
            sample = features[idx]
            util, sset = freddy(sample, K=30, alpha=alpha, beta=beta, return_vals=True)

            if not isinstance(prev, np.ndarray) or util.size == 0:
                prev = util
                continue
            curr_e = rmse(util, prev)
            # h = (np.linalg.norm(util - prev)) + 10e-6
            h += np.linalg.norm(util - prev)
            d += (curr_e - e) / h

            e = curr_e
            sset = idx[sset]
            w[sset] += 1
            prev = util
        print(f"[{__}] :: {e} ({alpha}, {beta})")
        _w = 1 - (w / w.sum())
        if abs(e) < tol:
            break
        features += (_w @ (features.T @ _w).T) + 10e-6
        loss.append(e)

        alpha += d * 10e-4
        beta += d * 10e-4
    # _w = 1 - (w / w.sum())
    # features = _w @ (dataset.T @ _w).T

    # exit()
    sset = freddy(features, K=K, alpha=alpha, beta=beta, batch_size=batch_size)
    # sset = kmeans_sampler(features, K=K)
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
