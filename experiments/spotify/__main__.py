import pandas as pd
from functools import partial
from xgboost import XGBRFRegressor
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from coreset.environ import load_config
from coreset.kmeans import kmeans_sampler
from coreset.utils import (
    random_sampler,
    hash_encoding,
    oht_coding,
    craig_baseline,
)
from coreset.lazzy_greed import lazy_greed
from coreset.evaluator import BaseExperiment, REPEAT


def norm_(dataset):
    names = dataset.columns
    return pd.DataFrame(maxabs_scale(dataset, axis=0), columns=names)


def column_apply(*funcs):
    def _clean_columns(dataset, /, *names):
        data = dataset.copy()
        imputer = SimpleImputer(fill_value=None, strategy="most_frequent")
        for fn, name in zip(funcs, names):
            data[name] = data[name].map(fn, na_action="ignore").astype(float)
        names = data.columns
        transformed = imputer.fit_transform(data)
        data = pd.DataFrame(transformed, columns=names)
        return data

    return _clean_columns


outfile, DATA_HOME, names, tgt_name = load_config()
dataset = pd.read_csv(DATA_HOME, engine="pyarrow")
max_size = int(len(dataset) * 0.8)
tgt_name = "streams"

clean_cols = column_apply(
    *(lambda x: (x.replace(",", "") if x.isnumeric() else None) for _ in range(3))
)

dataset = clean_cols(dataset, "in_deezer_playlists", "in_shazam_charts", "streams")


if __name__ == "__main__":
    # sampling strategies
    smpln = [
        partial(lazy_greed, K=int(max_size * 0.05), metric="codist"),
        partial(lazy_greed, K=int(max_size * 0.10), metric="codist"),
        partial(lazy_greed, K=int(max_size * 0.15), metric="codist"),
        partial(lazy_greed, K=int(max_size * 0.25), metric="codist"),
        partial(lazy_greed, K=int(max_size * 0.50), metric="codist"),
        random_sampler(n_samples=int(max_size * 0.05)),
        random_sampler(n_samples=int(max_size * 0.10)),
        random_sampler(n_samples=int(max_size * 0.15)),
        random_sampler(n_samples=int(max_size * 0.25)),
        random_sampler(n_samples=int(max_size * 0.50)),
        kmeans_sampler(K=int(max_size * 0.05)),
        kmeans_sampler(K=int(max_size * 0.10)),
        kmeans_sampler(K=int(max_size * 0.15)),
        kmeans_sampler(K=int(max_size * 0.25)),
        kmeans_sampler(K=int(max_size * 0.50)),
        craig_baseline(0.05),
        craig_baseline(0.10),
        craig_baseline(0.15),
        craig_baseline(0.25),
        craig_baseline(0.50),
    ]

    spotify = BaseExperiment(
        dataset, model=XGBRFRegressor, lbl_name=tgt_name, repeat=REPEAT
    )

    spotify.register_preprocessing(
        hash_encoding(
            "artist(s)_name",
            "track_name",
            n_features=7,
        ),
        oht_coding("key", "mode"),
        norm_,
    )

    spotify.register_metrics(mean_squared_error)

    for sampler in smpln:
        spotify(sampler=sampler)
    result = spotify()  # base de comparação
    result.to_csv(outfile, index=False)
