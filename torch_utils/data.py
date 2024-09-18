# [ok] teste - decorador de classes
# [ok] teste - random dataset
# [  ] refat - dataframe datase
# [  ] teste - dataframe datase


import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from .metrics import batched_euclidean
from coreset.craig.lazy_greedy import FacilityLocation, lazy_greedy_heap

__all__ = ["RandomTrainingSet", "CoresetRandomDataset", "PandasDataset"]


def _prepair_tensor(array):
    seq = np.array(array).astype(np.float64)
    seq = torch.from_numpy(seq).float()
    return seq


class AbstractDataset(ABC):

    def __init__(self) -> None:
        self._features = self.set_features()
        self._target = self.set_target()

    @abstractmethod
    def set_target(self):
        pass

    @abstractmethod
    def set_features(self):
        pass

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return _prepair_tensor(self._features[index]), _prepair_tensor(
            self._target[index]
        ).unsqueeze(-1)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target


##############################
# decoradores
##############################


def craig_dataset(dataset_class):
    class Coreset(dataset_class):
        finished = False

        def __init__(self, coreset_size=1, core_batch=1, *args, **kwargs) -> None:
            for k, v in kwargs.items():
                # print(f"{k}: {v}")
                self.__setattr__(k, v)

            self.core_batch = core_batch
            self.coreset_size = self.set_core_size(coreset_size)
            self._features = super().set_features()
            self._target = super().set_target()
            self.indices = self.set_indices(self._features)
            super().__init__(*args, **kwargs)
            # update_wrapper(self, dataset_class)

        def __getitem__(self, idx):
            return super().__getitem__(idx)

        @property
        def features(self):
            return super().features[self.indices]

        @property
        def target(self):
            return super().target[self.indices]

        def set_core_size(self, coreset_size):
            if isinstance(coreset_size, float):
                if coreset_size >= 1:
                    raise ValueError(
                        f"Float coreset_size only accepts values in [0, 1)]: {coreset_size}"
                    )
                return int(self.core_batch * coreset_size)
            return coreset_size

        def set_indices(self, features):
            idx = []
            indices = np.arange(len(features))
            if not self.finished:
                start = 0
                for D in self.batched_dist(features):
                    V = indices[start : start + self.core_batch].reshape(-1, 1)
                    B = self.coreset_size
                    start += self.core_batch

                    locator = FacilityLocation(D=D, V=V)
                    sset_idx, *_ = lazy_greedy_heap(F=locator, V=V, B=B)
                    idx += sset_idx
            self.finished = True
            return np.array(idx, dtype=int)

        def batched_dist(self, features):
            yield from batched_euclidean(features, self.core_batch)

    Coreset.__qualname__ = f"{dataset_class.__qualname__}Coreset"
    Coreset.__name__ = f"{dataset_class.__name__}Coreset"
    return Coreset


##############################
# fim
##############################


class RandomTrainingSet(AbstractDataset, Dataset):
    def __init__(self, size, n_features, scale=1):
        self.n_features = n_features
        self.size = int(size)
        self.scale = scale
        super().__init__()

    def set_indices(self):
        return np.arange(len(self._features), dtype=np.int32)

    def set_features(self):
        return np.linspace(
            (bound := np.ones(self.n_features) * self.scale) * -1,
            bound,
            self.size,
            axis=0,
        )

    def set_target(self):
        noize = np.random.normal(size=self._features.shape)
        return np.sum(np.sin(self._features) + noize, axis=-1)


class PandasDataset(Dataset):
    # colocar essa classe no padrão
    def __init__(self, dataframe, target_name):

        self._features = dataframe.drop(target_name, axis="columns")
        self._target = dataframe[target_name]

        self.indices = self._features.index.values

    def __getitem__(self, index):
        features = self._features.iloc[index]
        labels = self._target.iloc[index]
        features, labels = (
            torch.Tensor(features.values).float(),
            torch.Tensor([labels]).float(),
        )
        out = {"features": features, "targets": labels}
        return out


CraigRandomDataset = craig_dataset(RandomTrainingSet)
CraigPandasDataset = craig_dataset(PandasDataset)
