import os
import random
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

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


@TransformFunction(
    "native-country",
    "marital-status",
    "workclass",
    "occupation",
    "relationship",
    "education",
    "race",
    n_features=n_feat,
)
def hash_encoding(dataset, *hash_names, n_features=15):
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


@TransformFunction("sex")
def oht_coding(dataset, *names):
    return pd.get_dummies(dataset, columns=list(names), drop_first=False)


def split_dataset(dataset, test_size=0.2, label=tgt_name):
    train, test = train_test_split(dataset, test_size=test_size, shuffle=True)
    return (Dataset(train, label=label), Dataset(test, label=label))


@TransformFunction()
def norm_(dataset):
    y = dataset.pop(tgt_name).values
    names = dataset.columns
    out = pd.DataFrame(PCA(n_components=4).fit_transform(dataset))
    out[tgt_name] = y
    return out


# coding_task = pipeline(hash_encoding, oht_coding, norm_)
coding_task = pipeline(hash_encoding, oht_coding)

if __name__ == "__main__":

    dataset = pd.read_csv(train_pth, engine="pyarrow")
    dataset.columns = names
    dataset.replace(" ?", np.nan, inplace=True)
    dataset.dropna(axis="index")
    dataset.label = dataset.label.map({" >50K": 1, " <=50K": 0})

    # dataset = Dataset(dataset, tgt_name)
    dataset = coding_task(dataset)
    print(dataset.shape)
    result = []
    experiments = (split_dataset(dataset) for _ in range(n))
    for j, (i, (train_ds, test_ds)) in product(
        [0.01, 0.05, 0.1], enumerate(experiments)
    ):
        #########################################################################
        ################################ coreset ################################
        #########################################################################
        print(f"training {i} coreset[{j}]")
        sset = []
        idx = map(
            lambda L, w, alpha: lazy_greed(
                train_ds[train_ds.label == L],
                alpha=alpha,
                reduce_fn="mean",
                K=int(int(len(train_ds) * j) * w),
                batch_size=2000,
            )[0],
            [0, 1],
            [0.3, 0.7],
            [0.1, 4],
        )
        _start = datetime.now().timestamp()
        _ = [sset.extend(sub) for sub in idx]
        random.shuffle(sset)
        sset = train_ds[sset]
        core_boost = train_model(XGBClassifier(), sset)
        _stop = datetime.now().timestamp()
        for metric in [precision_score, f1_score, recall_score]:
            core_m = metric(test_ds.label, core_boost(test_ds), average=None)
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
        # experiments = (split_dataset(dataset) for _ in range(n))
        # for i, (train_ds, test_ds) in enumerate(experiments):
        start = datetime.now().timestamp()
        boost = train_model(XGBClassifier(), train_ds)
        stop = datetime.now().timestamp()
        for metric in [precision_score, f1_score, recall_score]:
            m_ = metric(test_ds.label, boost(test_ds), average=None)
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
