import os
import math
import pandas as pd
import numpy as np
from functools import partial
from pathlib import Path
from xgboost import XGBClassifier
from dotenv import load_dotenv

from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.experiments import Experiment
from coreset.utils import hash_encoding, oht_coding, random_sampler
from coreset.lazzy_greed import lazy_greed

load_dotenv()

DATA_HOME = os.environ.get("DATA_HOME")
DATA_HOME = Path(DATA_HOME, "adult")

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

train_pth = DATA_HOME / "adult.data"
MAX_SIZE = 26048
data = pd.read_csv(train_pth, engine="pyarrow")
data.columns = names
*names, tgt_name = names
data.replace(" ?", np.nan, inplace=True)
data.dropna(axis="index")
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})


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
        partial(lazy_greed, K=int(MAX_SIZE * 0.01), batch_size=32),
        partial(lazy_greed, K=int(MAX_SIZE * 0.02), batch_size=32),
        partial(lazy_greed, K=int(MAX_SIZE * 0.05), batch_size=32),
        partial(lazy_greed, K=int(MAX_SIZE * 0.10), batch_size=32),
        partial(lazy_greed, K=int(MAX_SIZE * 0.15), batch_size=32),
        partial(lazy_greed, K=int(MAX_SIZE * 0.25), batch_size=32),
    ]
    smpln = rgn_smpln + lazy_smpln

    adult = Experiment(data, model=XGBClassifier, lbl_name=tgt_name, repeat=1)

    adult.register_preprocessing(
        hash_encoding(
            "native-country", "occupation", "marital-status", "fnlwgt", n_features=15
        ),
        oht_coding("sex", "education", "race", "relationship", "workclass"),
    )

    adult.register_metrics(precision_score, recall_score, f1_score)

    for sampler in smpln:
        adult(sampler=sampler)
    result = adult()  # base de comparação
    print(result)
