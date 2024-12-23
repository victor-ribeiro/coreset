import pandas as pd
import numpy as np
from functools import partial
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import normalize

from freddy.evaluator import BaseExperiment, REPEAT
from freddy.lazzy_greed import freddy
from freddy.opt_freddy import opt_freddy
from freddy.dataset.utils import random_sampler, craig_baseline, transform_fn
from freddy.environ import load_config

outfile, DATA_HOME, names, tgt_name = load_config()

data = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
*names, tgt_name = names

data[tgt_name] = data[tgt_name] - 1

max_size = len(data) * 0.8
K = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.4]
if __name__ == "__main__":
    # sampling strategies
    # smpln = [freddy, opt_freddy, random_sampler, craig_baseline]
    smpln = [opt_freddy]

    covtype = BaseExperiment(
        data,
        model=XGBClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # covtype.register_preprocessing(transform_fn(normalize, tgt_name=tgt_name))

    covtype.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )
    for sampler in smpln:
        print(f"[{datetime.now()}] {sampler.__name__}")
        for k in K:
            covtype(sampler=partial(sampler, K=int(max_size * k)))
        print(f"[{datetime.now()}] {sampler.__name__}\t::\t OK")

    covtype()  # base de comparação
    result = covtype.metrics  # base de comparação

    result.to_csv(outfile, index=False)
