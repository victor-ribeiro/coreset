import unittest
import pandas as pd
import numpy as np

from freddy.dataset.dataset import Dataset


@lambda _: _()
def new_dataset():
    dataset = {
        "column_1": np.random.normal(0, 1, 10),
        "column_2": np.random.normal(0, 1, 10),
        "column_3": np.random.normal(0, 1, 10),
    }
    dataset = pd.DataFrame(dataset)
    return Dataset(dataset, name="test_dataset")


class TestIndexing(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dataset = new_dataset

    def test_dataset_int_indexing(self):
        shape = self.dataset[0].shape
        assert shape == (3,)

    def test_dataset_str_indexing(self):
        shape = self.dataset["column_1"].shape
        assert shape == (10,)

    def test_dataset_int_interval_indexing(self):
        shape = self.dataset[0:3].shape
        assert shape == (3, 3)

    def test_dataset_str_tuple_indexing(self):
        shape = self.dataset[["column_1", "column_2"]].shape
        assert shape == (10, 2)

    def test_dataset_str_tuple_int_interval_indexing(self):
        shape = self.dataset[0:3, ["column_1", "column_2"]].shape
        assert shape == (4, 2)
