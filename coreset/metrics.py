import numpy as np
from sklearn.metrics import pairwise_distances
from functools import lru_cache, cache
from itertools import batched
from numba import jit, njit

from coreset.dataset.dataset import Dataset

__all__ = ["METRICS"]

METRICS = {}


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
