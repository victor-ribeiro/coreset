from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Protocol, Dict, TypeVar
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


@dataclass(slots=True)
class Model:
    learner: Learner
    dataset: Dataset
    hyper: Dict = None
    _fited = False

    def __post_init__(self):
        self.hyper = self.learner.__dict__

    def __call__(self, X: Dataset, /) -> Any:
        return self.learner.predict(X._buffer)

    def fit(self):
        X_ = self.dataset._buffer.astype(float)
        y_ = self.dataset.label
        self.learner = self.learner.fit(X_, y_)


@timeit
def train_model(learner, dataset):
    model = Model(learner, dataset)
    model.fit()
    return model
