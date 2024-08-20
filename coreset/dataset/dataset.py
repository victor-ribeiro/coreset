import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, data=None, name="", label=None) -> None:
        self._buffer = data
        self.name = name
        self._buffer.index = range(len(self))

        self.label = self._buffer.pop(label).values if label else None

    def __len__(self):
        return len(self._buffer)

    def __iter__(self):
        buff = self._buffer.values
        if self.label:
            yield from zip(buff, self.label)
        else:
            yield from buff

    def __getitem__(self, idx):
        buff = None

        match idx:
            case list():
                try:
                    buff = self._buffer.iloc[idx].values
                except:
                    buff = self._buffer.loc[:, idx].values
            case str():
                buff = self._buffer.loc[:, idx].values
            case int():
                buff = self._buffer.iloc[idx, :].values
            case tuple():
                rols, cols = idx
                try:
                    buff = self._buffer.iloc[rols, cols].values
                except IndexError:
                    buff = self._buffer.loc[rols, cols].values
                except Exception as e:
                    return e
            case _:
                buff = self._buffer[idx].values

        return buff

    @property
    def columns(self):
        return self._buffer.columns.values.tolist()

    @property
    def empty(self):
        return bool(self._buffer)

    @property
    def index(self):
        return self._buffer.index.values.tolist()

    @property
    def size(self):
        return self.__len__()
