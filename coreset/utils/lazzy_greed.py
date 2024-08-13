###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################
import math
from typing import Callable
import numpy as np
from dataclasses import dataclass, field

from coreset.utils.dataset import Dataset
from coreset.utils.metrics import *


class Coreset(Dataset):
    def __init__(self, data=None, name="", k=1) -> None:
        super().__init__(data, name)
        self.S = np.zeros(k, dtype=int)
        self.score = np.zeros(len(self), dtype=float)

    @classmethod
    def max_norm(cls, sset):
        return sset.max(axis=1).sum()
        # return sset.max()


class FacilityLocation:
    def __init__(self, dist_fn="similarity", alpha=1) -> None:
        self.score = np.log(1 + alpha)
        self.alpha = alpha
        self.dist_fn = METRICS[dist_fn]
        self.max_lim = None

    def __call__(self, dataset: Dataset = None):
        pass

    def gain(self, dataset):
        if not np.any(dataset):
            return self.score
        max_ref = np.zeros(len(dataset))
        for d in self.dist_fn(dataset):
            max_norm = Coreset.max_norm(d)
            yield np.log(
                1 + (self.alpha / max_norm) * np.maximum(max_ref, d).sum(axis=1)
            ) * (1 / self.score) - self.score

    # def score(self):
    #     if not sset:
    #         return np.log(1 + self.alpha)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    ds = np.random.normal(0, 1, (100, 2))
    gain_fn = FacilityLocation()
    score = np.array([val for val in gain_fn(ds)]).flatten()
    score = np.sort(score, kind="heapsort")
    plt.plot(score)
    plt.show()
