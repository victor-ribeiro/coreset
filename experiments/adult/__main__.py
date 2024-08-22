import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier


from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.evaluator import BaseExperiment
from coreset.lazzy_greed import lazy_greed
from coreset.utils import hash_encoding, oht_coding, random_sampler
from coreset.environ import load_config

outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names
data.replace(" ?", np.nan, inplace=True)
data.dropna(axis="index")
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})

MAX_SIZE = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    rgn_smpln = [
        random_sampler(n_samples=int(MAX_SIZE * 0.01)),
        random_sampler(n_samples=int(MAX_SIZE * 0.02)),
        random_sampler(n_samples=int(MAX_SIZE * 0.05)),
        random_sampler(n_samples=int(MAX_SIZE * 0.10)),
        random_sampler(n_samples=int(MAX_SIZE * 0.15)),
        random_sampler(n_samples=int(MAX_SIZE * 0.25)),
    ]
    lazy_smpln = [
        partial(lazy_greed, K=int(MAX_SIZE * 0.01), batch_size=2000),
        partial(lazy_greed, K=int(MAX_SIZE * 0.02), batch_size=2000),
        partial(lazy_greed, K=int(MAX_SIZE * 0.05), batch_size=2000),
        partial(lazy_greed, K=int(MAX_SIZE * 0.10), batch_size=2000),
        partial(lazy_greed, K=int(MAX_SIZE * 0.15), batch_size=2000),
        partial(lazy_greed, K=int(MAX_SIZE * 0.25), batch_size=2000),
    ]
    smpln = rgn_smpln + lazy_smpln

    adult = BaseExperiment(data, model=XGBClassifier, lbl_name=tgt_name, repeat=1)

    adult.register_preprocessing(
        hash_encoding(
            "native-country", "occupation", "marital-status", "fnlwgt", n_features=15
        ),
        oht_coding("sex", "education", "race", "relationship", "workclass"),
    )

    adult.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    for sampler in smpln:
        adult(sampler=sampler)
    result = adult()  # base de comparação
    print(result)
