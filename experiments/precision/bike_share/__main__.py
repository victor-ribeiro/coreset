import pandas as pd
from functools import partial
from xgboost import XGBRegressor

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

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

# dataset[["atemp", "temp", "hum", "windspeed"]] = normalize(
#     dataset[["atemp", "temp", "hum", "windspeed"]]
# )

# dataset[["casual", "registered"]] = minmax_scale(dataset[["casual", "registered"]])


experiment, val = train_test_split(dataset, test_size=0.2)
y_val = val.pop(tgt_name)

# s_ = craig_baseline(int(len(X_train) * 0.05))(X_train)
# craig_model = regressor()
# craig_model.fit(
#     X_train.iloc[s_],
#     y_train.iloc[s_],
#     eval_set=[(X_train.iloc[s_], y_train.iloc[s_]), (val, y_val)],
#     verbose=False,
# )

for _ in range(50):

    X_train, X_test = train_test_split(experiment, test_size=0.3)
    X_train.index = range(len(X_train))
    X_test.index = range(len(X_test))

    y_train = X_train.pop(tgt_name)
    y_test = X_test.pop(tgt_name)

    model = regressor()
    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (val, y_val)], verbose=False
    )
    result = model.evals_result()

    sset = lazy_greed(dataset.values, K=int(len(X_train) * 0.1))
    X_sset = X_train.iloc[sset]
    y_sset = y_train.iloc[sset]

    sub_model = regressor()
    sub_model.fit(
        X_sset, y_sset, eval_set=[(X_sset, y_sset), (val, y_val)], verbose=False
    )
    plt.plot(
        sub_model.evals_result()["validation_1"]["rmse"],
        alpha=0.8,
        marker="_",
        c="g",
        linewidth=0,
    )

    r_sset = random_sampler(int(len(X_train) * 0.1))(X_train)
    X_smp = X_train.iloc[r_sset]
    y_smp = y_train.iloc[r_sset]

    rand_model = regressor()
    rand_model.fit(X_smp, y_smp, eval_set=[(X_smp, y_smp), (val, y_val)], verbose=False)
    plt.plot(
        rand_model.evals_result()["validation_1"]["rmse"],
        alpha=0.4,
        markersize=3,
        marker="_",
        linewidth=0,
        c="r",
    )


plt.plot(result["validation_1"]["rmse"], label="baseline", alpha=0.8)
# plt.plot(craig_model.evals_result()["validation_1"]["rmse"], label="craig", marker="_")

plt.legend()
plt.show()
