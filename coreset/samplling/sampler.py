from functools import singledispatchmethod
from collections.abc import Callable
from coreset.lazzy_greed import fastcore

METHOD = {"coreset": fastcore, "random": None}


class Sampler:

    def __init__(self, method: str = "random") -> None:
        self.method = METHOD[method]

    def _(self, method: Callable):
        pass
