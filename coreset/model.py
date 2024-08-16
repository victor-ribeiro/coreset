from enum import Enum, auto

_NAMES = ["CLASSIFICATION", "REGRESSION", "UNSUPERVISED"]


class TaskKind(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    UNSUPERVISED = auto()


class Model:
    def __init__(self, learner, **hparam) -> None:
        self.learner = learner(**hparam)
        self._fited = False
