import pandas as pd
import multiprocessing
import re
import pickle

from functools import partial
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import FastICA

from coreset.evaluator import BaseExperiment, TrainCurve, REPEAT
from coreset.lazzy_greed import lazy_greed
from coreset.utils import hash_encoding, oht_coding, random_sampler, craig_baseline
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config


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
data = CountVectorizer(max_features=1500).fit_transform(data).toarray()
data = FastICA(n_components=100).fit_transform(data)
data = pd.DataFrame(data=data)
data[tgt_name] = [*map(int, tgt)]
data[tgt_name] = data[tgt_name].map(lambda x: 1 if x > 5 else 0)
n_threads = int(multiprocessing.cpu_count() / 2)

max_size = len(data) * 0.8


if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.05)),
        partial(kmeans_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.05)),
        # craig_baseline(0.05),
    ]

    review = TrainCurve(
        data,
        task="binary_classification",
        model=partial(
            data,
            model=partial(
                XGBClassifier,
                eta=0.15,
                tree_method="hist",
                grow_policy="lossguide",
                n_estimators=500,
                early_stopping_rounds=10,
                nthread=n_threads,
                subsample=0.6,
                scale_pos_weight=1,
            ),
        ),
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    review.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    review()  # base de comparação
    for sampler in smpln:
        review(sampler=sampler)
    result = review.metrics  # base de comparação

    result.to_csv(outfile, index=False)
