import pandas as pd
from functools import partial
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

from coreset.environ import load_config
from coreset.utils import (
    craig_baseline,
    random_sampler,
    transform_fn,
    oht_coding,
)
from coreset.lazzy_greed import fastcore
from coreset.kmeans import kmeans_sampler
from coreset.evaluator import BaseExperiment, REPEAT
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, minmax_scale, OrdinalEncoder, maxabs_scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from functools import partial

outfile, DATA_HOME, names, tgt_name = load_config()

dataset = pd.read_csv(DATA_HOME, engine="pyarrow", index_col=0, skiprows=1)
max_size = len(dataset) * 0.8
names = dataset.columns

avg_names = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]

dataset[tgt_name] = dataset[avg_names].mean(axis=1)
dataset = dataset.drop(columns=avg_names)
# dataset = minmax_scale(dataset)

if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(craig_baseline, K=int(max_size * 0.01)),
        partial(craig_baseline, K=int(max_size * 0.02)),
        partial(craig_baseline, K=int(max_size * 0.03)),
        partial(craig_baseline, K=int(max_size * 0.04)),
        partial(craig_baseline, K=int(max_size * 0.05)),
        partial(craig_baseline, K=int(max_size * 0.10)),
        partial(craig_baseline, K=int(max_size * 0.15)),
        partial(craig_baseline, K=int(max_size * 0.25)),
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
    ]
    sgemm = BaseExperiment(
        dataset,
        model=XGBRegressor,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    sgemm.register_preprocessing(transform_fn(minmax_scale, tgt_name))

    sgemm.register_metrics(mean_squared_error)

    sgemm()  # base de comparação
    for sampler in smpln:
        sgemm(sampler=sampler)
    result = sgemm.metrics

    result.to_csv(outfile, index=False)
