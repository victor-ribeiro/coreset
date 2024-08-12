###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

from typing import Callable
import numpy as np
from dataclasses import dataclass, field

from coreset.utils.dataset import Dataset
from coreset.utils.metrics import *


class LazzyGreed:

    def __init__(self, similarity: str | Callable = "similarity") -> None:
        self.score = 0


# teste de cache
def marginal_utility(e, S=None):

    return np.log


# teste de cache
def facility_loc(S: np.array):
    return S.max(axis=0).sum()


def lazzy_greed(V, marginal_func, max_elemen=1):
    pass


if __name__ == "__main__":
    dataset = np.random.normal(0, 1, (100, 2))
    d = similarity(dataset, batch_size=32)
    for _ in d:
        print(_.shape)
        break
    # from sklearn.metrics import pairwise_distances
    # import numpy as np

    # dataset = np.random.normal(0, 1, (10, 2))
    # d = pairwise_distances(dataset)
    # print(d)
    # d -= d.max() * -1
    # print(np.diag(d))
