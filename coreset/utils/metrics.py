import numpy as np
from sklearn.metrics import pairwise_distances


# def similarity(pdist):
#     def inner(*args, **kwargs):
#         yield from [D.max() - D for D in pdist(*args, **kwargs)]

#     return inner


def pdist(dataset, metric="euclidean", batch_size=1):
    n = len(dataset)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch = dataset[start:end]
        yield pairwise_distances(dataset, batch, metric=metric)
