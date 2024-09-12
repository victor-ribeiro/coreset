import pandas as pd
from functools import partial
from xgboost import XGBRegressor

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

from coreset.environ import load_config
from coreset.utils import (
    random_sampler,
    hash_encoding,
    transform_fn,
    craig_baseline,
    oht_coding,
)
from coreset.lazzy_greed import lazy_greed
from coreset.kmeans import kmeans_sampler
from coreset.evaluator import BaseExperiment, REPEAT

import matplotlib.pyplot as plt

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
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from functools import partial


regressor = partial(
    XGBRegressor,
    booster="gblinear",
    enable_categorical=True,
    n_estimators=40,
)

dataset = pd.read_csv(DATA_HOME, names=names, engine="pyarrow", index_col=0, skiprows=1)
max_size = len(dataset) * 0.8
*names, _ = names

print(dataset.shape)

# preprocessing
encoder = oht_coding("workingday", "holiday", "weathersit")
dataset.drop(columns=["yr", "mnth", "hr", "dteday"], inplace=True)
dataset = encoder(dataset)
print(dataset.shape)

# ver funções de transformação
# dataset[["atemp", "temp", "hum", "windspeed"]] = normalize(
#     dataset[["atemp", "temp", "hum", "windspeed"]]
# )

# dataset[["casual", "registered"]] = minmax_scale(dataset[["casual", "registered"]])


if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.01), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.02), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.03), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.04), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.05), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.10), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.15), batch_size=256),
        partial(lazy_greed, K=int(max_size * 0.25), batch_size=256),
        random_sampler(n_samples=int(max_size * 0.01)),
        random_sampler(n_samples=int(max_size * 0.02)),
        random_sampler(n_samples=int(max_size * 0.03)),
        random_sampler(n_samples=int(max_size * 0.04)),
        random_sampler(n_samples=int(max_size * 0.05)),
        random_sampler(n_samples=int(max_size * 0.10)),
        random_sampler(n_samples=int(max_size * 0.15)),
        random_sampler(n_samples=int(max_size * 0.25)),
        craig_baseline(0.01),
        craig_baseline(0.02),
        craig_baseline(0.03),
        craig_baseline(0.04),
        craig_baseline(0.05),
        craig_baseline(0.10),
        craig_baseline(0.15),
        craig_baseline(0.25),
    ]
    bike_share = BaseExperiment(
        dataset, model=regressor, lbl_name=tgt_name, repeat=REPEAT
    )

    # ajustar aqui
    bike_share.register_preprocessing(
        hash_encoding("parents", "has_nurs", "form", n_features=10),
        transform_fn(encoding, tgt_name, *names[4:]),
    )

    bike_share.register_metrics(mean_squared_error)

    bike_share()  # base de comparação
    for sampler in smpln:
        bike_share(sampler=sampler)
    result = bike_share.metrics
    result.to_csv(outfile, index=False)
