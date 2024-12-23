import pandas as pd
from functools import partial
from datetime import datetime
from xgboost import XGBClassifier

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from freddy.environ import load_config
from freddy.dataset.utils import (
    random_sampler,
    hash_encoding,
    transform_fn,
    craig_baseline,
    oht_coding,
)
from freddy.lazzy_greed import freddy
from freddy.opt_freddy import opt_freddy
from freddy.evaluator import BaseExperiment, TrainCurve, REPEAT

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
    size = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    smpln = [opt_freddy, freddy, random_sampler, craig_baseline]
    nursery = BaseExperiment(
        dataset,
        model=XGBClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # teste com oht_coding. mudar voltar para hash_encoding
    nursery.register_preprocessing(
        transform_fn(encoding, tgt_name, *names[4:]),
        oht_coding("parents", "has_nurs", "form"),
        # transform_fn(encoding, tgt_name, *names),
    )

    nursery.register_metrics(
        partial(precision_score, average="macro"),
        partial(f1_score, average="macro"),
        partial(recall_score, average="macro"),
    )

    nursery()
    for sampler in smpln:
        print(f"[{datetime.now()}] {sampler.__name__}")
        for K in size:
            nursery(sampler=partial(sampler, K=int(max_size * K)))
        print(f"[{datetime.now()}] {sampler.__name__}\t::\t OK")

    result = nursery.metrics  # base de comparação
    result.to_csv(outfile, index=False)
