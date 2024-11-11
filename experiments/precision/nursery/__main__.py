import pandas as pd
from functools import partial
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config
from coreset.utils import (
    random_sampler,
    hash_encoding,
    transform_fn,
    craig_baseline,
    oht_coding,
)
from coreset.lazzy_greed import fastcore
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
        partial(fastcore, K=int(max_size * 0.01)),
        partial(fastcore, K=int(max_size * 0.02)),
        partial(fastcore, K=int(max_size * 0.03)),
        partial(fastcore, K=int(max_size * 0.04)),
        partial(fastcore, K=int(max_size * 0.05)),
        partial(fastcore, K=int(max_size * 0.10)),
        partial(fastcore, K=int(max_size * 0.15)),
        partial(fastcore, K=int(max_size * 0.25)),
        partial(random_sampler, K=int(max_size * 0.01)),
        partial(random_sampler, K=int(max_size * 0.02)),
        partial(random_sampler, K=int(max_size * 0.03)),
        partial(random_sampler, K=int(max_size * 0.04)),
        partial(random_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.10)),
        partial(random_sampler, K=int(max_size * 0.15)),
        partial(random_sampler, K=int(max_size * 0.25)),
        partial(craig_baseline, K=int(max_size * 0.01)),
        partial(craig_baseline, K=int(max_size * 0.02)),
        partial(craig_baseline, K=int(max_size * 0.03)),
        partial(craig_baseline, K=int(max_size * 0.04)),
        partial(craig_baseline, K=int(max_size * 0.05)),
        partial(craig_baseline, K=int(max_size * 0.10)),
        partial(craig_baseline, K=int(max_size * 0.15)),
        partial(craig_baseline, K=int(max_size * 0.25)),
    ]
    nursery = BaseExperiment(
        dataset,
        model=XGBClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # teste com oht_coding. mudar voltar para hash_encoding
    nursery.register_preprocessing(
        # hash_encoding("parents", "has_nurs", "form", n_features=5),
        oht_coding("parents", "has_nurs", "form"),
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
