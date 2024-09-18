from typing import Generator
import torch
from sklearn.metrics import pairwise_distances
from functools import partial


def batched_pairwise_dist(distance_func):
    def inner(dataset, batch_size=1):
        yield from (
            distance_func(dataset[start : start + batch_size], dataset)
            for start in range(0, len(dataset), batch_size)
        )

    return inner


batched_euclidean = partial(pairwise_distances, metric="euclidean")
batched_euclidean = batched_pairwise_dist(batched_euclidean)

batched_cosine = partial(pairwise_distances, metric="cosine")
batched_cosine = batched_pairwise_dist(batched_cosine)
