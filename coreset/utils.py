import re
import numpy as np
import pandas as pd
import multiprocessing as mp

from datetime import datetime
from itertools import batched
from functools import wraps
from unidecode import unidecode
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.feature_extraction.text import HashingVectorizer

from .craig.lazy_greedy import FacilityLocation, lazy_greedy_heap


N_JOBS = mp.cpu_count()


def timeit(f_):
    @wraps(f_)
    def inner(*args, **kwargs):
        print(f"[{datetime.now()}] {f_.__name__} :: ")
        start = datetime.now().timestamp()
        out = f_(*args, **kwargs)
        end = datetime.now().timestamp()
        print(f"[{datetime.now()}] {f_.__name__} :: {end - start:.4f}.S")
        return out

    return inner


def prepeair_encoding(data, *col_names):

    names = list(col_names)
    dataset = data.copy()
    cod_ds = dataset[names]
    cod_ds = cod_ds.astype(str).apply("-".join, axis=1)
    cod_ds = cod_ds.map(unidecode).str.lower()
    cod_ds = cod_ds.map(lambda x: re.sub("[{?$%&.;:(),}]", " ", x).replace(" ", "-"))
    dataset["coding"] = cod_ds.values
    return dataset.drop(columns=names)


def hash_encoding(*hash_names, n_features=15):
    @wraps(hash_encoding)
    def inner(data):
        encoded = prepeair_encoding(data, *hash_names)

        encoder = HashingVectorizer(n_features=n_features)
        encoded = encoder.fit_transform(encoded["coding"]).toarray()
        encoded = pd.DataFrame(encoded)
        dataset = pd.concat([data, encoded], axis="columns", join="inner")
        dataset = dataset.drop(columns=list(hash_names))
        dataset.columns = dataset.columns.astype(str)
        return dataset

    return inner


def oht_coding(*names):
    @wraps(oht_coding)
    def inner(dataset):
        return pd.get_dummies(dataset, columns=list(names), drop_first=False)

    return inner


def transform_fn(f_, tgt_name, *args, **kwargs):
    @wraps(f_)
    def inner(dataset):
        y = dataset.copy().pop(tgt_name).values
        names = dataset.columns
        data = f_(dataset, *args, **kwargs)
        out = pd.DataFrame(data, columns=names)
        out[tgt_name] = y
        return out

    return inner


def split_dataset(label, test_size=0.2):
    @wraps(split_dataset)
    def inner(dataset):
        train, test = train_test_split(
            dataset.copy(), test_size=test_size, shuffle=True
        )
        train_l, test_l = train.pop(label).values, test.pop(label).values
        return (train.values, train_l), (test.values, test_l)

    return inner


@timeit
def random_sampler(data, K):
    size = len(data)
    rng = np.random.default_rng()
    sset = rng.integers(0, size, size=K, dtype=int)
    return sset


# [v.03]
@timeit
def craig_baseline(data, K, b_size=1024):
    features = data.astype(np.single)
    V = np.arange(len(features), dtype=int).reshape(-1, 1)
    start = 0
    end = start + b_size
    sset = []
    n_jobs = int(N_JOBS // 2)
    for ds in batched(features, b_size):
        ds = np.array(ds)
        # D = pairwise_distances(features, ds, metric="euclidean", n_jobs=n_jobs)
        D = pairwise_distances(ds, features, metric="euclidean", n_jobs=n_jobs)
        v = V[start:end]
        D = D.max() - D
        B = int(len(D) * (K / len(features)))
        locator = FacilityLocation(D=D, V=v)
        sset_idx, *_ = lazy_greedy_heap(F=locator, V=v, B=B)
        sset_idx = np.array(sset_idx, dtype=int).reshape(1, -1)[0]
        sset.append(sset_idx)
        start += b_size
        end += b_size
    sset = np.hstack(sset)
    return sset


# [v.02]
# @timeit
# def craig_baseline(data, K):
#     features = data.astype(np.single)
#     start = 0
#     sset = []
#     # V = np.arange(len(features), dtype=int).reshape(-1, 1)
#     for D in pairwise_distances_chunked(features, metric="euclidean", n_jobs=3):
#         size = len(D)
#         V = np.arange(start, start + size, dtype=int).reshape(-1, 1)
#         D = D.max() - D
#         B = int(len(D) * (K / len(features)))
#         locator = FacilityLocation(D=D, V=V)
#         sset_idx, *_ = lazy_greedy_heap(F=locator, V=V, B=B)
#         sset_idx = np.array(sset_idx, dtype=int).reshape(1, -1)[0]
#         sset.append(sset_idx)
#         start += size
#     sset = np.hstack(sset)
#     print(f"Selected {len(sset)}")
#     return sset

# [v.01]
# @timeit
# def craig_baseline(data, K):
#     features = data.astype(np.single)
#     D = pairwise_distances(features, metric="euclidean", n_jobs=3)
#     D = D.max() - D
#     V = np.arange(len(features), dtype=int).reshape(-1, 1)
#     locator = FacilityLocation(D=D, V=V)
#     sset_idx, *_ = lazy_greedy_heap(F=locator, V=V, B=K)
#     sset_idx = np.array(sset_idx, dtype=int).reshape(1, -1)[0]
#     print(f"Selected {len(sset_idx)}")
#     return sset_idx
