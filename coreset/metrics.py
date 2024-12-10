import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from coreset.dataset.dataset import Dataset

__all__ = ["METRICS"]

METRICS = {}


def euclidean(data):
    x_ = np.sum(data**2, axis=1)
    y_ = np.sum(data**2, axis=1)[:, np.newaxis]
    dot = data @ data.T
    return np.sqrt(x_ + y_ - 2 * dot)


def _register(fn):
    name = fn.__name__
    METRICS[name] = fn
    __all__.append(name)
    return fn


@_register
def pdist(dataset, metric="euclidean", batch_size=1):
    return pairwise_distances(dataset, metric=metric)


@_register
def codist(dataset, batch_size=1):
    d = pairwise_distances(dataset, dataset, metric="cosine")
    return d.max() - d


@_register
def kdist(dataset, centroids):
    d = pairwise_distances(dataset, centroids)
    return d.max() - d


@_register
def similarity(dataset, metric="euclidean", batch_size=1):
    d = pdist(dataset, metric)
    return (d.max(axis=1) - d) / d.max(axis=1)


@_register
# def uni_similarity(dataset, batch_size=1):
# def similarity(dataset, batch_size=1):
#     d = pairwise_distances(dataset)
#     d /= d.sum(axis=0)
#     print(d.sum(axis=0))
#     exit()


@_register
def low_similarity(dataset, metric="euclidean", batch_size=1):
    pca = PCA(n_components=2)
    data = pca.fit_transform(dataset)
    d = pdist(data, metric, batch_size).astype(np.float16)
    return d.max() - d


if __name__ == "__main__":
    from sys import getsizeof
    from scipy.spatial.distance import cdist

    for i in np.linspace(1_000, 300_000, 5, dtype=int):
        data = np.random.normal(size=(i, 100))
        print(
            f"[DIST] matriz ({i}x{i}) :: {getsizeof(data)/8/1024:.2f}", end=" \t - \t"
        )
        d = cdist(data, data, metric="euclidean")
        print(f"{getsizeof(d) / 8 / 1024}:.2f")
        del data
        del d
