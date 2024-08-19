from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Protocol, Dict, TypeVar

from coreset.dataset.dataset import Dataset


class Learner(Protocol):
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

    def __call__(self, X, /) -> Any:
        return self.learner.predict(X)


def train_model(task: TaskKind):
    def deco(f_):
        def inner(model: Model, dataset: Dataset):
            match task:
                case TaskKind.CLASSIFICATION:
                    print("classificação")
                case TaskKind.REGRESSION:
                    print("regressão")
            return f_

        return inner

    return deco


@train_model(TaskKind.CLASSIFICATION)
def make_classification(f_):
    def inner(*args, kwargs):
        pass

    return inner


@train_model(TaskKind.CLASSIFICATION)
def make_regression(f_):
    def inner(*args, **kwargs):
        pass

    return inner


if __name__ == "__main__":
    pass
