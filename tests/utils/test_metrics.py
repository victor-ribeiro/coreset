from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

from coreset.utils.metrics import pdist
from coreset.utils.dataset import Dataset


@lambda _: _()
def new_dataset():
    return np.random.normal(0, 1, (10, 3))


def test_euclidean_distance():
    dataset: Dataset = new_dataset
    D = pairwise_distances(dataset)
    _D = np.column_stack([*pdist(dataset, batch_size=2)])
    assert np.allclose(D, _D)
