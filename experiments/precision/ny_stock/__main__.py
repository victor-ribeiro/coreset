import pandas as pd
from functools import partial
from xgboost import XGBClassifier

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config
from coreset.utils import random_sampler, hash_encoding, transform_fn, craig_baseline
from coreset.lazzy_greed import lazy_greed
from coreset.kmeans import kmeans_sampler
from coreset.evaluator import BaseExperiment, REPEAT

import matplotlib.pyplot as plt

outfile, DATA_HOME, names, tgt_name = load_config()


converters = {
    "open": float,
    "close": float,
    "low": float,
    "high": float,
    "volume": float,
}
dataset = pd.read_csv(DATA_HOME, names=names, converters=converters, skiprows=1)
max_size = len(dataset) * 0.8
names.remove(tgt_name)


dataset["date"] = pd.to_datetime(dataset["date"])
# dataset["open"] = pd.to_numeric(dataset["open"])
# dataset["close"] = pd.to_numeric(dataset["close"])
# dataset["low"] = pd.to_numeric(dataset["low"])
# dataset["high"] = pd.to_numeric(dataset["high"])
# dataset["volume"] = pd.to_numeric(dataset["volume"])


if __name__ == "__main__":
    print(dataset.nunique())
