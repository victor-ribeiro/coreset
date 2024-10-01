#### implementação usando o craig
# from coreset.craig.lazy_greedy import FacilityLocation, lazy_greedy_heap

# idx = []
#  np.arange(len(features))
# if not self.finished:
#     start = 0
#     for D in self.batched_dist(features):
#         V = indices[start : start + self.core_batch].reshape(-1, 1)
#         B = self.coreset_size
#         start += self.core_batch

#         locator = FacilityLocation(D=D, V=V)
#         sset_idx, *_ = lazy_greedy_heap(F=locator, V=V, B=B)
#         idx += sset_idx


import torch
import numpy as np
from torch.utils.data import Dataset

from .metrics import batched_euclidean

from coreset.kmeans import kmeans_sampler
from coreset.utils import random_sampler
from coreset.lazzy_greed import lazy_greed

# __all__ = ["RandomTrainingSet", "CoresetRandomDataset", "PandasDataset"]


def _prepair_tensor(array):
    seq = np.array(array).astype(np.float64)
    seq = torch.from_numpy(seq).float()
    return seq


class BaseDataset:

    __slots__ = ["_features", "_target", "_indices"]

    def __init__(self, features, target) -> None:
        self._features = features
        self._target = target
        self._indices = None
        self.set_indices()

    def __len__(self):
        return len(self._features)

    def __getitem__(self, index):
        return _prepair_tensor(self._features[index]), _prepair_tensor(
            self._target[index]
        )

    @property
    def shape(self):
        return len(self._features), len(self._target)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def set_indices(self):
        n, _ = self.shape
        self._indices = np.arange(n)


def sampling_dataset(dataset_class, sampler=None):
    class Coreset(dataset_class):

        __slots__ = ["coreset_size", "indices", "_features", "_target", "finished"]

        def __init__(self, features, target, coreset_size=1) -> None:
            self._features = features
            self._target = target
            self.coreset_size = coreset_size
            self.finished = False

            self.set_indices()

        @property
        def features(self):
            return super()._features[self.indices]

        @property
        def target(self):
            return super().target[self.indices]

        def set_indices(self):
            indices = sampler(self._features, K=self.coreset_size)
            self.finished = True
            return indices

    Coreset.__qualname__ = f"{dataset_class.__qualname__}Coreset"
    Coreset.__name__ = f"{dataset_class.__name__}Coreset"
    return Coreset
