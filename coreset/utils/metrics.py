import numpy as np
from sklearn.metrics import pairwise_distances

__all__ = ["METRICS"]

METRICS = {}


def _register(fn):
    name = fn.__name__
    METRICS[name] = fn
    __all__.append(name)
    return fn


def pdist(dataset, metric="euclidean", batch_size=1):
    n = len(dataset)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch = dataset[start:end]
        yield pairwise_distances(dataset, batch, metric=metric)


def similarity_fn(dataset, metric, batch_size=1):
    yield from [d.max() - d for d in pdist(dataset, metric, batch_size)]


_register(pdist)
_register(similarity_fn)
