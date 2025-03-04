import pandas as pd
from functools import partial
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config
from coreset.utils import random_sampler, hash_encoding, transform_fn, craig_baseline
from coreset.lazzy_greed import freddy
from coreset.kmeans import kmeans_sampler
from coreset.evaluator import BaseExperiment, TrainCurve, REPEAT

import matplotlib.pyplot as plt

outfile, DATA_HOME, names, tgt_name = load_config()
dataset = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
max_size = len(dataset) * 0.8
*names, tgt_name = names


def encoding(dataset, *columns):
    data = dataset.copy()
    for col in columns:
        data[col] = OrdinalEncoder().fit_transform(dataset[col].values.reshape(-1, 1))
    return data


dataset["children"] = dataset["children"].map({"1": 1, "2": 2, "3": 3, "more": 4})
print(dataset.shape)
dataset[tgt_name] = dataset[tgt_name].map(
    lambda x: "recommend" if x == "very_recom" or x == "priority" else x
)
dataset[tgt_name] = LabelEncoder().fit_transform(dataset[tgt_name]).astype(int)

if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(freddy, K=int(max_size * 0.01)),
        partial(freddy, K=int(max_size * 0.02)),
        partial(freddy, K=int(max_size * 0.03)),
        partial(freddy, K=int(max_size * 0.04)),
        partial(freddy, K=int(max_size * 0.05)),
        partial(freddy, K=int(max_size * 0.10)),
        partial(freddy, K=int(max_size * 0.15)),
        partial(freddy, K=int(max_size * 0.25)),
        partial(kmeans_sampler, K=int(max_size * 0.01)),
        partial(kmeans_sampler, K=int(max_size * 0.02)),
        partial(kmeans_sampler, K=int(max_size * 0.03)),
        partial(kmeans_sampler, K=int(max_size * 0.04)),
        partial(kmeans_sampler, K=int(max_size * 0.05)),
        partial(kmeans_sampler, K=int(max_size * 0.10)),
        partial(kmeans_sampler, K=int(max_size * 0.15)),
        partial(kmeans_sampler, K=int(max_size * 0.25)),
        partial(random_sampler, K=int(max_size * 0.01)),
        partial(random_sampler, K=int(max_size * 0.02)),
        partial(random_sampler, K=int(max_size * 0.03)),
        partial(random_sampler, K=int(max_size * 0.04)),
        partial(random_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.10)),
        partial(random_sampler, K=int(max_size * 0.15)),
        partial(random_sampler, K=int(max_size * 0.25)),
        # craig_baseline(0.01),
        # craig_baseline(0.02),
        # craig_baseline(0.03),
        # craig_baseline(0.04),
        # craig_baseline(0.05),
        # craig_baseline(0.10),
        # craig_baseline(0.15),
        # craig_baseline(0.25),
    ]
    nursery = BaseExperiment(
        dataset,
        model=DecisionTreeClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    nursery.register_preprocessing(
        hash_encoding("parents", "has_nurs", "form", n_features=5),
        transform_fn(encoding, tgt_name, *names[4:]),
    )

    nursery.register_metrics(
        partial(precision_score, average="macro"),
        partial(f1_score, average="macro"),
        partial(recall_score, average="macro"),
    )

    nursery()
    for sampler in smpln:
        nursery(sampler=sampler)
    result = nursery.metrics  # base de comparação
    result.to_csv(outfile, index=False)
