import os

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from dotenv import load_dotenv
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt


from coreset.dataset.dataset import Dataset
from coreset.dataset.transform import TransformFunction, pipeline
from coreset.model import train_model
from coreset.lazzy_greed import lazy_greed

load_dotenv()

DATA_HOME = os.environ.get("DATA_HOME")
DATA_HOME = Path(DATA_HOME, "adult")

train_pth = DATA_HOME / "adult.data"

n_feat = 8
n = 10
names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "label",
]
tgt_name = names[-1]


def hash_encoding(*hash_names, n_features=15):
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
    def inner(dataset):
        return pd.get_dummies(dataset, columns=list(names), drop_first=False)

    return inner


def split_dataset(dataset, test_size=0.2, label=tgt_name):
    train, test = train_test_split(
        dataset, test_size=test_size, shuffle=True, stratify=dataset[label]
    )
    train_l, test_l = train.pop(label).values, test.pop(label).values
    return (train.values, train_l), (test.values, test_l)


def norm_(dataset):
    y = dataset.pop(tgt_name).values
    names = dataset.columns
    out = pd.DataFrame(PCA(n_components=4).fit_transform(dataset))
    out[tgt_name] = y
    return out


# coding_task = pipeline(hash_encoding, oht_coding, norm_)
coding_task = pipeline(
    hash_encoding(
        "native-country",
        "marital-status",
        "workclass",
        "occupation",
        "relationship",
        "education",
        "race",
        "age",
        "fnlwgt",
        n_features=n_feat,
    ),
    oht_coding("sex"),
)

if __name__ == "__main__":

    dataset = pd.read_csv(train_pth, engine="pyarrow")
    dataset.columns = names
    dataset.replace(" ?", np.nan, inplace=True)
    dataset.dropna(axis="index")
    dataset[tgt_name] = dataset.label.map({" >50K": 1, " <=50K": 0})

    # dataset = Dataset(dataset, tgt_name)
    dataset = coding_task(dataset).astype(float)

    result = []
    experiments = (split_dataset(dataset) for _ in range(n))
    for j, (i, (train, test)) in product([0.01, 0.05, 0.1], enumerate(experiments)):
        #########################################################################
        ################################ coreset ################################
        #########################################################################
        train_ds, train_label = train
        test_ds, test_label = test
        train_ds = normalize(train_ds, axis=0)
        test_ds = normalize(test_ds, axis=0)

        print(f"training {i} coreset[{j}] : {len(train_ds)} :")
        sset = []
        idx = map(
            lambda L: lazy_greed(
                train_ds[train_label == L],
                alpha=1,
                reduce_fn="sum",
                metric="similarity",
                K=int(int(len(train_ds) * j) / 2),
                batch_size=2000,
            )[0],
            [0, 1],  # label
        )
        _start = datetime.now().timestamp()
        [sset.extend(sub) for sub in idx]
        core_boost = XGBClassifier()
        core_boost.fit(train_ds[sset], train_label[sset])

        _stop = datetime.now().timestamp()
        for metric in [precision_score, f1_score, recall_score]:
            core_m = metric(test_label, core_boost.predict(test_ds), average=None)
            core_m = dict(zip([0, 1], core_m))
            core_m["metric"] = metric.__name__
            core_m["coreset"] = j
            core_m["size"] = len(sset)
            core_m["elapsed"] = _stop - _start
            result.append(core_m)
        print(f"training {i} coreset[{j}] : {len(sset)} : {_stop - _start:.4f}.S")

        #########################################################################
        ############################### baseline ################################
        #########################################################################
        start = datetime.now().timestamp()
        boost = XGBClassifier()
        boost.fit(train_ds, train_label)
        stop = datetime.now().timestamp()
        for metric in [precision_score, f1_score, recall_score]:
            m_ = metric(test_label, boost.predict(test_ds), average=None)
            m_ = dict(zip([0, 1], m_))
            m_["metric"] = metric.__name__
            m_["coreset"] = 1
            m_["size"] = len(dataset)
            m_["elapsed"] = stop - start
            result.append(m_)
        print(f"training {i} coreset[1] : {len(dataset)} : {stop - start:.4f}.S")

    result = pd.DataFrame.from_records(result)
    sns.scatterplot(data=result, x="elapsed", y="coreset")
    plt.yscale("log")
    plt.show()
    result.to_csv("experiments/adult/adult.csv", index=False)
    result = pd.melt(result, ["metric", "coreset", "size", "elapsed"])

    sns.catplot(
        data=result, y="value", col="variable", hue="metric", x="coreset", kind="box"
    )
    plt.show()
