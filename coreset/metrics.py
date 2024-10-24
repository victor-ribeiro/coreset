import numpy as np
from sklearn.metrics import pairwise_distances
from functools import lru_cache, cache
from itertools import batched
from numba import jit, njit

from coreset.dataset.dataset import Dataset

__all__ = ["METRICS"]

METRICS = {}


def mat_to_bytes(nrows, ncols, dtype=32, out="GB"):
    """Calculate the size of a numpy array in bytes.
    :param nrows: the number of rows of the matrix.
    :param ncols: the number of columns of the matrix.
    :param dtype: the size of each element in the matrix. Defaults to 32bits.
    :param out: the output unit. Defaults to gigabytes (GB)
    :returns: the size of the matrix in the given unit
    :rtype: a float
    """
    sizes = {v: i for i, v in enumerate("BYTES KB MB GB TB".split())}
    return nrows * ncols * dtype / 8 / 1024.0 ** sizes[out]


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


def pdist(dataset, metric="euclidean", batch_size=1):
    return pairwise_distances(dataset, metric=metric)


def codist(dataset, batch_size=1):
    d = pairwise_distances(dataset, dataset, metric="cosine")
    return d.max() - d


def similarity(dataset, metric="euclidean", batch_size=1):
    # yield from (d.max() - d for d in pdist(dataset, metric, batch_size))
    d = pdist(dataset, metric, batch_size)
    return d.max() - d


_register(pdist)
_register(similarity)
_register(codist)


if __name__ == "__main__":
    from sys import getsizeof

    for i in np.linspace(1_000, 300_000, 5, dtype=int):
        data = np.random.normal(size=(i, 100))
        print(
            f"[DIST] matriz ({i}x{i}) :: {getsizeof(data)/8/1024:.2f}", end=" \t - \t"
        )
        d = euclidean(data)
        print(f"{getsizeof(d) / 8 / 1024}:.2f")
        del data
        del d
