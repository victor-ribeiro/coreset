import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier


from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.evaluator import BaseExperiment, TrainCurve, REPEAT
from coreset.lazzy_greed import lazy_greed
from coreset.utils import hash_encoding, oht_coding, random_sampler, craig_baseline
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config

outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names
data.replace(" ?", np.nan, inplace=True)
# data.dropna(axis="index")
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})

max_size = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.05)),
        partial(kmeans_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.05)),
        # craig_baseline(0.05),
    ]

    adult = TrainCurve(
        data,
        task="binary_classification",
        model=partial(
            XGBClassifier,
            enable_categorical=True,
            grow_policy="lossguide",
        ),
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    adult.register_preprocessing(
        hash_encoding(
            "native-country", "occupation", "marital-status", "fnlwgt", n_features=5
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
