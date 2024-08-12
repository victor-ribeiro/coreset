###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################
import math
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
    import matplotlib.pyplot as plt

    dataset = np.random.normal(0, 1, (100, 2))
    d = similarity(dataset, batch_size=1)
    score = np.zeros(len(dataset))
    aux = 0
    for i, _ in enumerate(d):
        inc = (
            (np.log(1 + 1 / _.max(axis=1).sum() * _.sum(axis=1)) - score[i])
            * 1
            / score[i]
        )
        # inc = np.log(1 + 1  * _.sum(axis=1)) - aux
        score[i] = max(inc, score[i])
    # inc *= 1 / inc
    # score = np.sort(1 / score, kind="heapsort")
    score = np.sort(score, kind="heapsort")
    # plt.plot(1 / score * score, marker="o")
    plt.plot(score)
    plt.show()
    # print(f"{inc=}{aux=}")
    # aux = max(inc, aux)
    # from sklearn.metrics import pairwise_distances
    # import numpy as np

    # dataset = np.random.normal(0, 1, (10, 2))
    # d = pairwise_distances(dataset)
    # print(d)
    # d -= d.max() * -1
    # print(np.diag(d))
