import os
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
from dotenv import load_dotenv
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from coreset.dataset.dataset import Dataset
from coreset.lazzy_greed import lazy_greed

load_dotenv()

DATA_HOME = os.environ.get("DATA_HOME")
DATA_HOME = Path(DATA_HOME, "adult")


def pipeline(*f_):
    def inner(dataset):
        result = dataset
        for fn in f_:
            result = fn(result)
        return result

    return inner


def transform_wrap(data, func, tgt_name):
    dataset = data.copy()
    tgt = dataset[tgt_name].astype(int)
    names = dataset.columns.astype(str)
    dataset = pd.DataFrame(func(dataset), columns=names)
    dataset[tgt_name] = tgt.values
    return dataset


def hash_encoding(*hash_names, n_features=15):
    def inner(data):
        dataset = data.copy()
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


if __name__ == "__main__":

    train_pth = DATA_HOME / "adult.data"
    test_pth = DATA_HOME / "adult.test"
    n_feat = 100
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

    # function seting
    preprocessing = pipeline(
        hash_encoding(
            "native-country",
            "marital-status",
            "workclass",
            "occupation",
            "relationship",
            "race",
            n_features=n_feat,
        ),
        partial(pd.get_dummies, columns=["sex"], drop_first=False),
        # minmax_scale
    )
    # data_load and split
    dataset = pd.read_csv(train_pth, engine="pyarrow")
    dataset.columns = names
    dataset.drop(columns="education", inplace=True)
    # dataset.replace(" ?", np.nan, inplace=True)
    dataset.replace(" ?", 0, inplace=True)
    # dataset.dropna(axis="index", inplace=True)
    dataset.label = dataset.label.map({" >50K": 1, " <=50K": 0})
    dataset = preprocessing(dataset)
    result = []
    for i in range(n):
        X_train, X_test = train_test_split(
            dataset, test_size=0.2, stratify=dataset.label.values
        )

        y_train = X_train.pop(tgt_name)
        y_test = X_test.pop(tgt_name)

        #########################################################################
        ############################### baseline ################################
        #########################################################################
        xgboost = XGBClassifier()
        start = datetime.now().timestamp()
        xgboost.fit(X_train, y_train)
        stop = datetime.now().timestamp()
        pred = xgboost.predict(X_test)
        for metric in [precision_score, f1_score, recall_score]:
            m_ = metric(y_test, pred, average=None)
            m_ = dict(zip([0, 1], m_))
            m_["metric"] = metric.__name__
            m_["coreset"] = 1
            m_["elapsed"] = stop - start
            result.append(m_)
        #########################################################################
        ################################ coreset ################################
        #########################################################################
        for j in [0.01, 0.05, 0.1, 0.15, 20, 50]:
            sset = []
            _start = datetime.now().timestamp()

            for w in [0, 1]:
                idx, _ = lazy_greed(
                    Dataset(X_train[(y_train == w)]),
                    alpha=1,
                    K=int(len(X_train) * j / 2),
                    batch_size=32,
                    reduce_fn="mean",
                )
                sset += idx

            core_boost = XGBClassifier()
            core_boost.fit(X_train.iloc[sset], y_train.iloc[sset])
            _stop = datetime.now().timestamp()
            pred = core_boost.predict(X_test)
            for metric in [precision_score, f1_score, recall_score]:
                m_ = metric(y_test, pred, average=None)
                m_ = dict(zip([0, 1], m_))
                m_["metric"] = metric.__name__
                m_["coreset"] = j
                m_["elapsed"] = _stop - _start
                result.append(m_)

    result = pd.DataFrame.from_records(result)
    sns.lineplot(data=result, y="elapsed", x="coreset")
    plt.show()
    result.to_csv("experiments/adult.csv", index=False)
    result = pd.melt(result, ["metric", "coreset", "elapsed"])

    sns.catplot(
        data=result, y="value", col="variable", hue="metric", x="coreset", kind="box"
    )
    plt.show()
