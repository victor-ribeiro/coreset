import pandas as pd
from functools import partial
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config
from coreset.utils import random_sampler, hash_encoding, transform_fn
from coreset.lazzy_greed import lazy_greed
from coreset.evaluator import BaseExperiment, REPEAT


outfile, DATA_HOME, names, tgt_name = load_config()
dataset = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
max_size = len(dataset) * 0.8
*names, tgt_name = names


def encoding(dataset, *columns):
    data = dataset.copy()
    for col in columns:
        data[col] = OrdinalEncoder().fit_transform(dataset[col].values.reshape(-1, 1))
    return data


dataset["children"] = dataset["children"].map({"1": 1, "2": 2, "3": 3, "more": 4})
dataset[tgt_name] = dataset[tgt_name].map(
    lambda x: "recommend" if x == "very_recom" or x == "priority" else x
)
dataset[tgt_name] = LabelEncoder().fit_transform(dataset[tgt_name]).astype(int)

if __name__ == "__main__":
    # sampling strategies
    rgn_smpln = [
        random_sampler(n_samples=int(max_size * 0.005)),
        random_sampler(n_samples=int(max_size * 0.01)),
        random_sampler(n_samples=int(max_size * 0.02)),
        random_sampler(n_samples=int(max_size * 0.05)),
        random_sampler(n_samples=int(max_size * 0.10)),
        random_sampler(n_samples=int(max_size * 0.15)),
        random_sampler(n_samples=int(max_size * 0.25)),
    ]
    lazy_smpln = [
        partial(lazy_greed, K=int(max_size * 0.005), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.01), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.02), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.05), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.10), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.15), batch_size=64),
        partial(lazy_greed, K=int(max_size * 0.25), batch_size=64),
    ]
    smpln = rgn_smpln + lazy_smpln
    nursery = BaseExperiment(
        dataset, model=XGBClassifier, lbl_name=tgt_name, repeat=REPEAT
    )

    nursery.register_preprocessing(
        hash_encoding("parents", "has_nurs", "form", n_features=10),
        transform_fn(encoding, tgt_name, *names[4:]),
    )

    nursery.register_metrics(
        partial(precision_score, average="macro"),
        partial(recall_score, average="macro"),
        partial(f1_score, average="macro"),
    )

    for sampler in smpln:
        nursery(sampler=sampler)
    result = nursery()  # base de comparação
    result.to_csv(outfile, index=False)
