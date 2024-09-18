from abc import ABC, abstractmethod
from datetime import datetime
from inspect import signature

from typing import Dict, Any, Callable


class BaseTask(ABC):
    def __init__(self):
        self._start_time: datetime = datetime.now()
        self._end_time: datetime = None
        self._meta: Dict[str, Any] = {}

    @abstractmethod
    def __call__(self):
        pass

    @property
    def finished(self):
        return bool(self._end_time)


def callable_task(cls):

    class NewTask(cls, BaseTask):
        _meta = {}

        def __init__(self, func, /):
            super().__init__(func)
            self.func = func
            self.iter = iter
            self.set_metainfo()

        def set_metainfo(self):
            sig = signature(self.func)
            self._meta["name_fn"] = self.func.__name__
            self._meta["parameters"] = dict(sig.parameters)
            self._meta["return_annot"] = sig.return_annotation

            if doc := self.func.__doc__:
                self._meta["doc"] = doc

    return NewTask


@callable_task
class TrainTask:
    # implementar o treinamento aqui com as funções e um dicionário com histórico de métricas
    _name: str = "train"
    _epochs: int = 1

    def __init__(self, epochs=1):
        self._epochs = epochs
