import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

from functools import partial
from itertools import product

from coreset.environ import load_config
from coreset.utils import transform_fn, random_sampler
from coreset.lazzy_greed import freddy
from coreset.opt_freddy import opt_freddy
from coreset.evaluator import BSizeExperiment, REPEAT


outfile, DATA_HOME, names, tgt_name = load_config()

dataset = pd.read_csv(DATA_HOME, engine="pyarrow", index_col=0, skiprows=1)
max_size = len(dataset) * 0.8
names = dataset.columns

avg_names = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]

dataset[tgt_name] = dataset[avg_names].mean(axis=1)
dataset = dataset.drop(columns=avg_names)
# dataset = minmax_scale(dataset)

b_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

if __name__ == "__main__":
    # sampling strategies
    sgemm = BSizeExperiment(
        dataset,
        model=XGBRegressor,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # sgemm.register_preprocessing(transform_fn(minmax_scale, tgt_name))

    sgemm.register_metrics(mean_squared_error)
    sgemm()  # base de comparação
    for K in [0.1, 0.25, 0.4]:
        for size in b_size:
            sgemm(
                sampler=partial(freddy, K=int(max_size * K)),
                batch_size=size,
            )
            sgemm(
                sampler=partial(opt_freddy, K=int(max_size * K)),
                batch_size=size,
            )
    result = sgemm.metrics

    result.to_csv(outfile, index=False)
