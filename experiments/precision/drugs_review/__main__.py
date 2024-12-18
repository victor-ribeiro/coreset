import re
import multiprocessing
import pickle
import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, f1_score, recall_score

from xgboost import XGBClassifier

from coreset.lazzy_greed import freddy
from coreset.opt_freddy import opt_freddy
from coreset.utils import random_sampler, craig_baseline
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config
from coreset.evaluator import BaseExperiment, REPEAT


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    # sent = " ".join(sent).lower()
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent


outfile, DATA_HOME, names, tgt_name = load_config()

with open(DATA_HOME, "rb") as file:

    data = pickle.load(file)

data, tgt = data["features"], data["target"]
data = map(clean_sent, data)

# data = CountVectorizer(max_features=1500).fit_transform(data).toarray()
data = (
    TfidfVectorizer(max_features=1500, min_df=0.05, max_df=0.98)
    .fit_transform(data)
    .toarray()
)
data = PCA(n_components=100).fit_transform(data)
data = pd.DataFrame(data=data)
data[tgt_name] = [*map(int, tgt)]
data[tgt_name] = data[tgt_name].map(lambda x: 1 if x > 5 else 0)
data.columns = data.columns.astype(str)

max_size = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    smpln = [freddy, random_sampler, craig_baseline]
    size = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n_threads = int(multiprocessing.cpu_count() / 2)

    review = BaseExperiment(
        data,
        model=partial(XGBClassifier, eta=0.15, max_depth=9, nthread=n_threads),
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    review.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )
    for sampler in smpln:
        for K in size:
            print(f"[{datetime.now()}] {sampler.__name__} ({K})")
            review(sampler=partial(sampler, K=int(max_size * K)))
        print(f"[{datetime.now()}] {sampler.__name__}\t::\t OK")
    review()  # base de comparação
    result = review.metrics  # base de comparação

    result.to_csv(outfile, index=False)
