import numpy as np
from sklearn.metrics import pairwise_distances

METRICS = {}


def _register(fn):
    def inner(*args, **kwargs):
        METRICS[fn.__name__] = fn

    return fn


@_register
def pdist(dataset, metric="euclidean", batch_size=1):
    n = len(dataset)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch = dataset[start:end]
        yield pairwise_distances(dataset, batch, metric=metric)
