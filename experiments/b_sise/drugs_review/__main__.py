import re
import multiprocessing
import pickle
import numpy as np
import pandas as pd

from functools import partial
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, f1_score, recall_score

from xgboost import XGBClassifier

from coreset.lazzy_greed import freddy
from coreset.opt_freddy import opt_freddy
from coreset.utils import random_sampler
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config
from coreset.evaluator import BSizeExperiment, REPEAT


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    # sent = " ".join(sent).lower()
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent


outfile, DATA_HOME, names, tgt_name = load_config()

with open(DATA_HOME, "rb") as file:

    dataset = pickle.load(file)

dataset, tgt = dataset["features"], dataset["target"]
dataset = map(clean_sent, dataset)

dataset = CountVectorizer(max_features=1500).fit_transform(dataset).toarray()
dataset = PCA(n_components=100).fit_transform(dataset)
dataset = pd.DataFrame(data=dataset)
dataset[tgt_name] = [*map(int, tgt)]
dataset[tgt_name] = dataset[tgt_name].map(lambda x: 1 if x > 5 else 0)
dataset.columns = dataset.columns.astype(str)

max_size = len(dataset) * 0.8
b_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

if __name__ == "__main__":
    # sampling strategies

    n_threads = int(multiprocessing.cpu_count() / 2)

    review = BSizeExperiment(
        dataset,
        model=XGBClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    review.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    review()  # base de comparação
    for K in [0.1, 0.25, 0.40]:
        for size in b_size:
            review(
                sampler=partial(freddy, K=int(max_size * K)),
                batch_size=size,
            )
            review(
                sampler=partial(opt_freddy, K=int(max_size * K)),
                batch_size=size,
            )

    result = review.metrics

    result.to_csv(outfile, index=False)
