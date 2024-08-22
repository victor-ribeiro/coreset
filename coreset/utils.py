import re
import pandas as pd
from datetime import datetime
import random
from functools import wraps
from unidecode import unidecode

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer


def timeit(f_):
    @wraps(f_)
    def inner(*args, **kwargs):
        start = datetime.now().timestamp()
        out = f_(*args, **kwargs)
        end = datetime.now().timestamp()
        print(f"[RUNNING] {f_.__name__} :: {end - start:.4f}.S")
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


def random_sampler(n_samples):
    @timeit
    @wraps(random_sampler)
    def inner(data):

        size = len(data)
        sset = range(size)
        sset = random.choices(sset, k=n_samples)
        return sset

    return inner
