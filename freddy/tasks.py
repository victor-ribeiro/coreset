# TODO: Implement the Task class and its subclasses
# implementar tasks para classificação e regressão
# especializar classes para NLP (exemplo drugs review)

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any


class Task(ABC):
    _name = None
    __slots__ = (
        "train_fn",
        "model",
        "data",
        "metrics",
        "epochs",
        "_start_time",
        "_end_time",
    )

    def __init__(self, train_fn, model, data, metrics, epochs):
        self.train_fn = train_fn
        self.model = model
        self.data = data
        self.metrics = metrics
        self.epochs = epochs
        self._start_time = None
        self._end_time = None

    @abstractmethod
    def __call__(self):
        pass

    @property
    def finished(self):
        return bool(self._end_time)


class ClassificationTask(Task):
    _name = "classification"

    def __init__(self, train_fn, model, data, metrics, epochs):
        super().__init__(train_fn, model, data, metrics, epochs)

    def __call__(self) -> Dict[str, Any]:
        self._start_time = datetime.now()
        for _ in range(self.epochs):
            self.train_fn(self.model, self.data)
        self._end_time = datetime.now()
        return self.metrics
