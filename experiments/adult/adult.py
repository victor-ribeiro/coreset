import os
import pandas as pd
import numpy as np
from functools import partial
from pathlib import Path
from xgboost import XGBClassifier
from dotenv import load_dotenv

from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import minmax_scale
from coreset.experiments import Experiment
from coreset.utils import hash_encoding, transform_fn, oht_coding, split_dataset
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
data = pd.read_csv(train_pth, engine="pyarrow")
data.columns = names
*names, tgt_name = names

data.replace(" ?", np.nan, inplace=True)
data.dropna(axis="index")
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})

adult = Experiment(data, model=XGBClassifier, lbl_name=tgt_name)

adult.register_preprocessing(
    hash_encoding(
        "native-country", "occupation", "marital-status", "fnlwgt", n_features=8
    ),
    oht_coding("sex", "education", "race", "relationship", "workclass"),
)

adult.register_metrics(precision_score, recall_score, f1_score)


for ex in [200, 500, 1000, 5000, 10000]:
    result = adult(repeat=2, sampler=partial(lazy_greed, K=ex, batch_size=512))
print(result)
