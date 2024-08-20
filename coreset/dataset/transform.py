import numpy as np
import pandas as pd

from coreset.dataset.dataset import Dataset


def make_dataset(label=None):
    def inner(data):
        buff = pd.DataFrame(data)
        return Dataset(buff, label=label)

    return inner


class TransformFunction:

    def __init__(self, func, *args, **kwargs) -> None:

        self.func = func
        self.named_args = args
        self.unnamed_args = kwargs

    def __call__(self, dataset):
        return self.func(dataset, *self.named_args, **self.unnamed_args)


def pipeline(*chain: TransformFunction):
    def inner(dataset):
        result = dataset._buffer
        for func in chain:
            result = func(result)
        return result

    return inner
