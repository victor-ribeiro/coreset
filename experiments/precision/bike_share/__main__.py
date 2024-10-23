import pandas as pd
from functools import partial
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

from coreset.environ import load_config
from coreset.utils import (
    random_sampler,
    transform_fn,
    craig_baseline,
    oht_coding,
)
from coreset.lazzy_greed import lazy_greed
from coreset.kmeans import kmeans_sampler
from coreset.evaluator import BaseExperiment, REPEAT


outfile, DATA_HOME, names, tgt_name = load_config()


#####################################################################################
###                              :: PREPROCEDING ::                               ###
###   DROP: instant, yr, mnth, hr, dteday                                         ###
###   ONE HOT ENCODING: workingday(0, 1), holiday(0, 1), weathersit(1,2,3,4) ::   ###
###   KEEP: season, weekday                                                       ###
###   NORMALIZE: atemp, temp, hum, windspeed                                      ###
###   MIN_MAX: casual, registered                                                 ###
#####################################################################################

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, minmax_scale, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from functools import partial

dataset = pd.read_csv(DATA_HOME, names=names, engine="pyarrow", index_col=0, skiprows=1)
max_size = len(dataset) * 0.8
*names, _ = names

# preprocessing
# encoder = oht_coding("workingday", "holiday", "weathersit")
dataset["dteday"] = OrdinalEncoder().fit_transform(
    dataset["dteday"].values.reshape(-1, 1)
)
# dataset = encoder(dataset)


def normal_cols(dataset, *names):
    names = [*names]
    aux_ds = dataset.copy()
    aux_ds[names] = normalize(aux_ds[names])
    return aux_ds


def scale_cols(dataset, *names):
    names = [*names]
    aux_ds = dataset.copy()
    aux_ds[names] = minmax_scale(aux_ds[names])
    return aux_ds


if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.01)),
        partial(lazy_greed, K=int(max_size * 0.02)),
        partial(lazy_greed, K=int(max_size * 0.03)),
        partial(lazy_greed, K=int(max_size * 0.04)),
        partial(lazy_greed, K=int(max_size * 0.05)),
        partial(lazy_greed, K=int(max_size * 0.10)),
        partial(lazy_greed, K=int(max_size * 0.15)),
        partial(lazy_greed, K=int(max_size * 0.25)),
        partial(kmeans_sampler, K=int(max_size * 0.01)),
        partial(kmeans_sampler, K=int(max_size * 0.02)),
        partial(kmeans_sampler, K=int(max_size * 0.03)),
        partial(kmeans_sampler, K=int(max_size * 0.04)),
        partial(kmeans_sampler, K=int(max_size * 0.05)),
        partial(kmeans_sampler, K=int(max_size * 0.10)),
        partial(kmeans_sampler, K=int(max_size * 0.15)),
        partial(kmeans_sampler, K=int(max_size * 0.25)),
        partial(random_sampler, K=int(max_size * 0.01)),
        partial(random_sampler, K=int(max_size * 0.02)),
        partial(random_sampler, K=int(max_size * 0.03)),
        partial(random_sampler, K=int(max_size * 0.04)),
        partial(random_sampler, K=int(max_size * 0.05)),
        partial(random_sampler, K=int(max_size * 0.10)),
        partial(random_sampler, K=int(max_size * 0.15)),
        partial(random_sampler, K=int(max_size * 0.25)),
        partial(craig_baseline, K=0.01),
        partial(craig_baseline, K=0.02),
        partial(craig_baseline, K=0.03),
        partial(craig_baseline, K=0.04),
        partial(craig_baseline, K=0.05),
        partial(craig_baseline, K=0.10),
        partial(craig_baseline, K=0.15),
        partial(craig_baseline, K=0.25),
    ]
    bike_share = BaseExperiment(
        dataset,
        model=DecisionTreeClassifier,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # ajustar aqui
    bike_share.register_preprocessing(
        oht_coding("workingday", "holiday", "weathersit"),
        transform_fn(normal_cols, tgt_name, "atemp", "temp", "hum", "windspeed"),
        transform_fn(scale_cols, tgt_name, "casual", "registered"),
    )

    bike_share.register_metrics(mean_squared_error)

    bike_share()  # base de comparação
    for sampler in smpln:
        bike_share(sampler=sampler)
    result = bike_share.metrics

    result.to_csv(outfile, index=False)
