###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

from typing import Callable
import numpy as np
from dataclasses import dataclass, field

from coreset.utils.dataset import Dataset
from coreset.utils.metrics import pdist


class Coreset(Dataset):
    def __init__(self, data=None, name="", alpha=1.0) -> None:
        super().__init__(data, name)

        self.idx = np.arange((n := len(self)))
        self.score = np.zeros(n)

    def similatiry(self):
        pass


# teste de cache
def marginal_utility(S, e):
    return np.log


# teste de cache
def facility_loc(S: np.array):
    return S.max(axis=0).sum()


def lazzy_greed(V, marginal_func, max_elemen=1):
    pass


if __name__ == "__main__":
    from sklearn.metrics import pairwise_distances
    import numpy as np

    dataset = np.random.normal(0, 1, (10, 2))
    d = pairwise_distances(dataset)
    print(d)
    d -= d.max() * -1
    print(np.diag(d))
