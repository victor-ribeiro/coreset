from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

from coreset.utils.metrics import pdist
from coreset.utils.dataset import Dataset


@lambda _: _()
def new_dataset():
    return np.random.normal(0, 1, (10, 3))


def test_custom_dataset_euclidean_distance():
    dataset = pd.DataFrame(new_dataset)
    D = pairwise_distances(dataset)
    _D = pdist(Dataset(dataset), batch_size=2)
    _D = np.vstack([*_D])
    assert np.allclose(D, _D) and D.shape == _D.shape
