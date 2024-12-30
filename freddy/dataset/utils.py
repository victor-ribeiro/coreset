import re
import pandas as pd
from unidecode import unidecode

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer


def prepeair_encoding(data, *col_names):
    names = list(col_names)
    dataset = data.copy()
    cod_ds = dataset[names]
    cod_ds = cod_ds.astype(str).apply("-".join, axis=1)
    cod_ds = cod_ds.map(unidecode).str.lower()
    cod_ds = cod_ds.map(lambda x: re.sub("[{?$%&.;:(),}]", " ", x).replace(" ", "-"))
    dataset["coding"] = cod_ds.values
    return dataset.drop(columns=names)


def hash_encoding(data, *hash_names, n_features=15):

    encoded = prepeair_encoding(data, *hash_names)
    encoder = HashingVectorizer(n_features=n_features)
    encoded = encoder.fit_transform(encoded["coding"]).toarray()
    encoded = pd.DataFrame(encoded)
    dataset = pd.concat([data, encoded], axis="columns", join="inner")
    dataset = dataset.drop(columns=list(hash_names))
    dataset.columns = dataset.columns.astype(str)
    return dataset


def split_dataset(dataset, label, test_size=0.2):
    train, test = train_test_split(dataset.copy(), test_size=test_size, shuffle=True)
    train_l, test_l = train.pop(label).values, test.pop(label).values
    return (train.values, train_l), (test.values, test_l)
