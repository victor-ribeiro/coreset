from enum import Enum, auto
from typing import Any, Protocol, Dict
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from coreset.dataset.dataset import Dataset
from coreset.utils import timeit
import numpy as np


class Learner(Protocol):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class TaskKind(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    UNSUPERVISED = auto()


class Backend(Enum):
    TORCH = auto()
    SKLEARN = auto()


class FitError(Exception):
    def __init__(self) -> None:
        super().__init__("Model instance not fited")


class BaseLearner(ABC):
    __slots__ = ["learner", "hyper", "_fited", "_model"]

    def __init__(self, learner, hyper) -> None:

        self.learner = learner
        self.hyper = hyper
        self._model = learner(**hyper)
        self._fited = False

    def __post_init__(self):
        self.hyper = self.learner.__dict__

    @abstractmethod
    def __call__(self, X: Dataset, /) -> Any:
        pass

    @property
    def fited(self):
        return self._fited

    @fited.setter
    def fited(self, fit):
        self._fited = fit


class SklearnLearner(BaseLearner):
    _backend = Backend.SKLEARN

    def __call__(self, X: Dataset) -> Any:
        if not self._fited:
            return FitError
        return self._model.predict(X)

    def fit(self, X, y, *args, **kwargs):
        try:
            model = self._model.fit(X, y, *args, **kwargs)
            self._fited = True
            return model
        except Exception as e:
            return e


class TorchLearner(BaseLearner):
    def __call__(self, X: Dataset) -> Any:
        import torch

        if not self._fited:
            return FitError
        ft = torch.Tensor(X)
        pred = self._model(ft)
        return pred.detach().numpy()
