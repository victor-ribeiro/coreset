import pandas as pd


class Dataset:

    def __init__(self, data=None, name="") -> None:
        self._buffer = data
        self.name = name

    def __len__(self):
        return len(self._buffer)

    def __iter__(self):
        yield from self._buffer.values

    def __getitem__(self, idx):
        buff = None

        match idx:
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
        return self._buffer.index.values

    @property
    def size(self):
        return self.__len__()
