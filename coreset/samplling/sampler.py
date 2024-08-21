from functools import singledispatchmethod
from collections.abc import Callable
from coreset.lazzy_greed import lazy_greed

METHOD = {"coreset": lazy_greed, "random": None}


class Sampler:

    def __init__(self, method: str = "random") -> None:
        self.method = METHOD[method]

    def _(self, method: Callable):
        pass
