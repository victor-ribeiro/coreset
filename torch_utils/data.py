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
from functools import partial

from freddy.kmeans import kmeans_sampler
from freddy.dataset.utils import random_sampler
from freddy.lazzy_greed import freddy

SAMPLING = {
    "kmeans_sampler": kmeans_sampler,
    "random_sampler": random_sampler,
    "lazy_greed": freddy,
}

# __all__ = ["RandomTrainingSet", "CoresetRandomDataset", "PandasDataset"]


def _prepair_tensor(array, dtype=torch.FloatTensor):
    seq = np.array(array).astype(np.float32)
    seq = torch.from_numpy(seq).type(dtype)
    return seq


def sampling_dataset(dataset_class, sampler=None, dtype=torch.FloatTensor):
    class Coreset(dataset_class):

        __slots__ = [
            "coreset_size",
            "_indices",
            "_features",
            "_target",
            "finished",
            "_sampler",
        ]

        def __init__(self, features, target, coreset_size=1, dtype=dtype) -> None:
            target = np.array(target)
            indices = sampler(features, K=coreset_size)
            self._features = features[indices]
            self._target = target[indices]
            self.coreset_size = coreset_size
            self.dtype = dtype
            self._sampler = sampler
            self.finished = False

    Coreset.__qualname__ = f"Coreset{dataset_class.__qualname__}"
    Coreset.__name__ = f"Coreset{dataset_class.__name__}"
    return Coreset


class BaseDataset:

    __slots__ = ["_features", "_target", "_indices", "dtype"]

    def __init__(self, features, target, dtype=torch.FloatTensor) -> None:
        self._features = features
        self._target = target
        self.dtype = dtype
        self._indices = None
        self.set_indices()

    def __len__(self):
        return len(self._features)

    def __getitem__(self, index):
        return _prepair_tensor(self._features[index], self.dtype), _prepair_tensor(
            self._target[index], self.dtype
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
