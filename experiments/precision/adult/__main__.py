import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.evaluator import BaseExperiment, REPEAT
from coreset.lazzy_greed import freddy
from coreset.utils import hash_encoding, oht_coding, random_sampler, craig_baseline
from coreset.kmeans import kmeans_sampler
from coreset.environ import load_config


outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names
data.replace(" ?", np.nan, inplace=True)
data[tgt_name] = data.label.map({" >50K": 1, " <=50K": 0})

max_size = len(data) * 0.8

if __name__ == "__main__":
    # sampling strategies
    size = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    smpln = [craig_baseline, freddy, random_sampler]

    adult = BaseExperiment(
        data,
        # model=partial(XGBClassifier, device="gpu"),
        model=partial(XGBClassifier),
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    adult.register_preprocessing(
        hash_encoding("native-country", n_features=5),
        hash_encoding("occupation", n_features=5),
        hash_encoding("marital-status", n_features=5),
        # hash_encoding("native-country", "occupation", "marital-status", n_features=5),
        oht_coding("sex", "education", "race", "relationship", "workclass"),
    )

    adult.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    adult()  # base de comparação
    for sampler in smpln:
        print(f"[{datetime.now()}] {sampler.__name__}")
        for K in size:
            adult(sampler=partial(sampler, K=int(max_size * K)))
        print(f"[{datetime.now()}] {sampler.__name__}\t::\t OK")

    result = adult.metrics  # base de comparação

    result.to_csv(outfile, index=False)
