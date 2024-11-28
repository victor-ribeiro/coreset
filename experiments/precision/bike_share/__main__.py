import pandas as pd
from functools import partial
from itertools import batched
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

from coreset.environ import load_config
from coreset.utils import random_sampler, craig_baseline, oht_coding, transform_fn
from coreset.lazzy_greed import freddy
from coreset.opt_freddy import opt_freddy
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

from sklearn.preprocessing import normalize, minmax_scale, OrdinalEncoder
from functools import partial

dataset = pd.read_csv(DATA_HOME, names=names, engine="pyarrow", index_col=0, skiprows=1)
max_size = len(dataset) * 0.8
*names, _ = names
dataset["dteday"] = OrdinalEncoder().fit_transform(
    dataset["dteday"].values.reshape(-1, 1)
)


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
    size = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    smpln = [opt_freddy, freddy, random_sampler, craig_baseline]
    # smpln = [freddy]
    bike_share = BaseExperiment(
        dataset,
        model=XGBRegressor,
        lbl_name=tgt_name,
        repeat=REPEAT,
    )

    # ajustar aqui
    bike_share.register_preprocessing(
        oht_coding("workingday", "holiday", "weathersit"),
        transform_fn(scale_cols, tgt_name, "casual", "registered"),
        # transform_fn(normal_cols, tgt_name, "casual", "registered"),
    )

    bike_share.register_metrics(mean_squared_error)

    print(f"[{datetime.now()}] BASELINE")
    bike_share()  # base de comparação
    for sampler in smpln:
        print(f"[{datetime.now()}] {sampler.__name__}")
        for K in size:
            bike_share(sampler=partial(sampler, K=int(max_size * K)))
        print(f"[{datetime.now()}] {sampler.__name__}\t::\t OK")

    result = bike_share.metrics

    result.to_csv(outfile, index=False)
