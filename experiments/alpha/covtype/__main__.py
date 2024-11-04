import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import normalize

from coreset.evaluator import BaseExperiment, REPEAT
from coreset.lazzy_greed import fastcore
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
        partial(fastcore, K=int(max_size * 0.01)),
        partial(fastcore, K=int(max_size * 0.02)),
        partial(fastcore, K=int(max_size * 0.03)),
        partial(fastcore, K=int(max_size * 0.04)),
        partial(fastcore, K=int(max_size * 0.05)),
        partial(fastcore, K=int(max_size * 0.10)),
        partial(fastcore, K=int(max_size * 0.15)),
        partial(fastcore, K=int(max_size * 0.25)),
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

    covtype = BaseExperiment(
        data,
        model=DecisionTreeClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    covtype.register_preprocessing(transform_fn(normalize, tgt_name=tgt_name))

    covtype.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    covtype()  # base de comparação
    for sampler in smpln:
        covtype(sampler=sampler)
    result = covtype.metrics  # base de comparação

    result.to_csv(outfile, index=False)
