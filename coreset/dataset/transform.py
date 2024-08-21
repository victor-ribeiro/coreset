import pandas as pd
from sklearn.model_selection import train_test_split
from functools import singledispatch
from coreset.dataset.dataset import Dataset
from coreset.utils import timeit
import numpy as np


def make_dataset(label=None):
    def inner(f_, /, *args, **kwargs):
        def deco(data):
            buff = pd.DataFrame(data)
            buff = Dataset(buff, label=label)
            return f_(buff, *args, **kwargs)

        return deco

    return inner


class TransformFunction:
    __slots__ = ("func", "named_args", "unnamed_args")

    def __init__(self, *args, **kwargs) -> None:

        self.func = None
        self.named_args = args
        self.unnamed_args = kwargs

    def __call__(self, f_):
        def inner(dataset):
            self.func = f_
            return self.func(dataset, *self.named_args, **self.unnamed_args)

        return inner


def pipeline(*chain: TransformFunction, label=None):
    @make_dataset(label=label)
    @timeit
    def inner(dataset):
        result = dataset._buffer
        for func in chain:
            result = func(result)
        return result

    return inner
