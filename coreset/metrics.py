import numpy as np
from sklearn.metrics import pairwise_distances
from functools import lru_cache, cache
from itertools import batched


__all__ = ["METRICS"]

METRICS = {}


def _register(fn):
    name = fn.__name__
    METRICS[name] = fn
    __all__.append(name)
    return fn


def pdist(dataset, metric="euclidean", batch_size=1):
    n = len(dataset)
    yield from (
        pairwise_distances(batch, dataset, metric=metric)
        for batch in batched(dataset, batch_size)
    )


def similarity(dataset, metric="euclidean", batch_size=1):
    yield from [d.max() - d for d in pdist(dataset, metric, batch_size)]


_register(pdist)
_register(similarity)
