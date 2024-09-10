import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier


from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.evaluator import BaseExperiment, REPEAT
from coreset.lazzy_greed import lazy_greed
from coreset.utils import hash_encoding, oht_coding, random_sampler, craig_baseline
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config

outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names
data.replace(" ?", np.nan, inplace=True)
data.dropna(axis="index")
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})

max_size = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.01), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.02), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.03), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.04), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.05), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.10), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.15), batch_size=1024),
        partial(lazy_greed, K=int(max_size * 0.25), batch_size=1024),
        random_sampler(n_samples=int(max_size * 0.01)),
        random_sampler(n_samples=int(max_size * 0.02)),
        random_sampler(n_samples=int(max_size * 0.03)),
        random_sampler(n_samples=int(max_size * 0.04)),
        random_sampler(n_samples=int(max_size * 0.05)),
        random_sampler(n_samples=int(max_size * 0.10)),
        random_sampler(n_samples=int(max_size * 0.15)),
        random_sampler(n_samples=int(max_size * 0.25)),
        craig_baseline(0.01),
        craig_baseline(0.02),
        craig_baseline(0.03),
        craig_baseline(0.04),
        craig_baseline(0.05),
        craig_baseline(0.10),
        craig_baseline(0.15),
        craig_baseline(0.25),
    ]

    adult = BaseExperiment(data, model=XGBClassifier, lbl_name=tgt_name, repeat=REPEAT)

    adult.register_preprocessing(
        hash_encoding(
            "native-country", "occupation", "marital-status", "fnlwgt", n_features=10
        ),
        oht_coding("sex", "education", "race", "relationship", "workclass"),
    )

    adult.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    adult()  # base de comparação
    for sampler in smpln:
        adult(sampler=sampler)
    result = adult.metrics  # base de comparação

    result.to_csv(outfile, index=False)