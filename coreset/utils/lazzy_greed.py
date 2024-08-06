from coreset.utils.dataset import Dataset
import numpy as np


def facility_location(subset: np.array):
    return subset.max(axis=1).sum()


def lazzygreed():
    pass


def submodular():
    pass
