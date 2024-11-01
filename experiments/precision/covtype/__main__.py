import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import normalize

from coreset.evaluator import BaseExperiment, REPEAT
from coreset.lazzy_greed import lazy_greed, lazy_greed_class
from coreset.utils import (
    hash_encoding,
    oht_coding,
    random_sampler,
    craig_baseline,
    transform_fn,
)
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config


outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names

data[tgt_name] = data[tgt_name] - 1

max_size = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.01)),
        partial(lazy_greed, K=int(max_size * 0.02)),
        partial(lazy_greed, K=int(max_size * 0.03)),
        partial(lazy_greed, K=int(max_size * 0.04)),
        partial(lazy_greed, K=int(max_size * 0.05)),
        partial(lazy_greed, K=int(max_size * 0.10)),
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

    covtype = BaseExperiment(
        data,
        model=XGBClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    covtype.register_preprocessing(transform_fn(normalize, tgt_name=tgt_name))

    covtype.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    for sampler in smpln:
        # covtype(sampler=sampler, task="classification")
        covtype(sampler=sampler)
    covtype()  # base de comparação
    result = covtype.metrics  # base de comparação

    result.to_csv(outfile, index=False)
