from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Protocol, Dict

from coreset.dataset import Dataset

_NAMES = ["CLASSIFICATION", "REGRESSION", "UNSUPERVISED"]


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


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = Model(HistGradientBoostingClassifier(max_bins=20), None)
    print(model)
