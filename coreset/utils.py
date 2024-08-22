import pandas as pd
from datetime import datetime
import random
from functools import wraps

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


def hash_encoding(*hash_names, n_features=15):
    @wraps(hash_encoding)
    def inner(dataset):
        corpus = (
            dataset[list(hash_names)]
            .apply(
                lambda x: "".join(str(x)),
                axis="columns",
            )
            .values
        )

        encoder = HashingVectorizer(n_features=n_features)
        encoded = encoder.fit_transform(corpus).toarray()
        encoded = pd.DataFrame(encoded)
        dataset = pd.concat([dataset, encoded], axis="columns", join="inner")
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
        y = dataset.pop(tgt_name).values
        names = dataset.columns
        data = f_(dataset, *args, **kwargs)
        out = pd.DataFrame(data, columns=names)
        out[tgt_name] = y
        return out

    return inner


def split_dataset(label, test_size=0.2):
    @wraps(split_dataset)
    def inner(dataset):
        train, test = train_test_split(dataset, test_size=test_size, shuffle=True)
        train_l, test_l = train.pop(label).values, test.pop(label).values
        return (train.values, train_l), (test.values, test_l)

    return inner


def random_sampler(n_samples, random_state=42):
    @timeit
    @wraps(random_sampler)
    def inner(data):
        random.seed(random_state)

        size = len(data)
        sset = range(size)
        sset = random.choices(sset, k=n_samples)
        return sset

    return inner
