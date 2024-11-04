import pandas as pd
import numpy as np
from functools import partial
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import precision_score, f1_score, recall_score

from coreset.evaluator import BaseExperiment, REPEAT
from coreset.lazzy_greed import fastcore
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
    smpln = [
        partial(fastcore, K=int(max_size * 0.01)),
        partial(fastcore, K=int(max_size * 0.02)),
        partial(fastcore, K=int(max_size * 0.03)),
        partial(fastcore, K=int(max_size * 0.04)),
        partial(fastcore, K=int(max_size * 0.05)),
        partial(fastcore, K=int(max_size * 0.10)),
        partial(fastcore, K=int(max_size * 0.15)),
        partial(fastcore, K=int(max_size * 0.25)),
        partial(random_sampler, K=int(max_size * 0.01)),
        partial(random_sampler, K=int(max_size * 0.02)),
        partial(random_sampler, K=int(max_size * 0.03)),
        partial(random_sampler, K=int(max_size * 0.04)),
        partial(random_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.10)),
        partial(random_sampler, K=int(max_size * 0.15)),
        partial(random_sampler, K=int(max_size * 0.25)),
        partial(craig_baseline, K=int(max_size * 0.01)),
        partial(craig_baseline, K=int(max_size * 0.02)),
        partial(craig_baseline, K=int(max_size * 0.03)),
        partial(craig_baseline, K=int(max_size * 0.04)),
        partial(craig_baseline, K=int(max_size * 0.05)),
        partial(craig_baseline, K=int(max_size * 0.10)),
        partial(craig_baseline, K=int(max_size * 0.15)),
        partial(craig_baseline, K=int(max_size * 0.25)),
    ]

    adult = BaseExperiment(
        data,
        model=partial(XGBClassifier, device="gpu"),
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    ### datasets baharam : minist e cifar
    ### matrix de distâncias (mapa de calor)

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
