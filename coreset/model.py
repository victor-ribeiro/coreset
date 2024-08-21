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
    label: list
    hyper: Dict = None
    _fited = False

    def __post_init__(self):
        self.hyper = self.learner.__dict__

    def __call__(self, X: Dataset, /) -> Any:
        return self.learner.predict(X)

    def fit(self):
        X_ = self.dataset.values.astype(float)
        y_ = self.label
        self.learner = self.learner.fit(X_, y_)


@timeit
def train_model(learner, dataset, label):
    model = Model(learner, dataset, label)
    model.fit()
    return model
