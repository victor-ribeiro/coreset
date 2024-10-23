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


if __name__ == "__main__":
    from nearpy import Engine
    from nearpy.hashes import RandomBinaryProjections

    # Dimension of our vector space
    dimension = 500

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections("rbp", 10)

    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp])

    # Index 1000000 random vectors (set their data to a unique string)
    for index in range(100000):
        v = np.random.randn(dimension)
        engine.store_vector(v, "data_%d" % index)

    # Create random query vector
    query = np.random.randn(dimension)

    # Get nearest neighbours
    N = engine.neighbours(query)
