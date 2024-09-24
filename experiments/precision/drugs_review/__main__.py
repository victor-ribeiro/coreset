# import pandas as pd
# from functools import partial
# from xgboost import XGBClassifier

# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config

# from coreset.utils import random_sampler, hash_encoding, transform_fn, craig_baseline
# from coreset.lazzy_greed import lazy_greed
# from coreset.kmeans import kmeans_sampler
# from coreset.evaluator import BaseExperiment, TrainCurve, REPEAT

# import matplotlib.pyplot as plt

outfile, DATA_HOME, names, tgt_name = load_config()
# dataset = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
# max_size = len(dataset) * 0.8
# *names, tgt_name = names


# def encoding(dataset, *columns):
#     data = dataset.copy()
#     for col in columns:
#         data[col] = OrdinalEncoder().fit_transform(dataset[col].values.reshape(-1, 1))
#     return data


# dataset["children"] = dataset["children"].map({"1": 1, "2": 2, "3": 3, "more": 4})
# print(dataset.shape)
# dataset[tgt_name] = dataset[tgt_name].map(
#     lambda x: "recommend" if x == "very_recom" or x == "priority" else x
# )
# dataset[tgt_name] = LabelEncoder().fit_transform(dataset[tgt_name]).astype(int)

# if __name__ == "__main__":
#     # sampling strategies
#     smpln = [
#         partial(lazy_greed, K=int(max_size * 0.01), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.02), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.03), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.04), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.05), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.10), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.15), batch_size=256),
#         partial(lazy_greed, K=int(max_size * 0.25), batch_size=256),
#         kmeans_sampler(K=int(max_size * 0.01)),
#         kmeans_sampler(K=int(max_size * 0.02)),
#         kmeans_sampler(K=int(max_size * 0.03)),
#         kmeans_sampler(K=int(max_size * 0.04)),
#         kmeans_sampler(K=int(max_size * 0.05)),
#         kmeans_sampler(K=int(max_size * 0.10)),
#         kmeans_sampler(K=int(max_size * 0.15)),
#         kmeans_sampler(K=int(max_size * 0.25)),
#         random_sampler(n_samples=int(max_size * 0.01)),
#         random_sampler(n_samples=int(max_size * 0.02)),
#         random_sampler(n_samples=int(max_size * 0.03)),
#         random_sampler(n_samples=int(max_size * 0.04)),
#         random_sampler(n_samples=int(max_size * 0.05)),
#         random_sampler(n_samples=int(max_size * 0.10)),
#         random_sampler(n_samples=int(max_size * 0.15)),
#         random_sampler(n_samples=int(max_size * 0.25)),
#         craig_baseline(0.01),
#         craig_baseline(0.02),
#         craig_baseline(0.03),
#         craig_baseline(0.04),
#         craig_baseline(0.05),
#         craig_baseline(0.10),
#         craig_baseline(0.15),
#         craig_baseline(0.25),
#     ]
#     nursery = BaseExperiment(
#         dataset,
#         model=partial(
#             XGBClassifier,
#             enable_categorical=True,
#             grow_policy="lossguide",
#             n_estimators=30,
#         ),
#         lbl_name=tgt_name,
#         repeat=REPEAT,
#     )

#     nursery.register_preprocessing(
#         hash_encoding("parents", "has_nurs", "form", n_features=5),
#         transform_fn(encoding, tgt_name, *names[4:]),
#     )

#     nursery.register_metrics(
#         partial(precision_score, average="macro"),
#     )

#     nursery()
#     for sampler in smpln:
#         nursery(sampler=sampler)
#     result = nursery.metrics  # base de comparação
#     result.to_csv(outfile, index=False)

##### TODO
## [ ] vetorizar
## [ ] treinar modelo
## [ ] template avaliação de precisão e curva de aprendizado
## [ ] _make_dataset.py_ para a pasta data
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    FeatureHasher,
    HashingVectorizer,
)
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab")

import multiprocessing

with open(DATA_HOME, "rb") as file:
    data = pickle.load(file)

n_threads = multiprocessing.cpu_count()
min_df = 0.03
max_df = 1 - min_df

X_train, y_train = data["features"], data["target"]
# X_train = TfidfVectorizer(
#     max_df=max_df, min_df=min_df, stop_words="english", max_features=200
# ).fit_transform(X_train)
# X_train = CountVectorizer(
#     max_df=max_df, min_df=min_df, stop_words="english", max_features=200
# ).fit_transform(X_train)
# X_train = map(word_tokenize, X_train)

X_train = np.array(X_train)
X_train = (
    # FeatureHasher(n_features=300, input_type="string").transform(X_train).toarray()
    HashingVectorizer(n_features=300).fit_transform(X_train)
)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

y_train = np.array(y_train)
y_test = np.array(y_test)

# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize, quantile_transform, minmax_scale

# x, y = PCA(n_components=2).fit_transform(X_train).T
# plt.scatter(x, y, c=y_train)
# plt.title(X_train.shape)
# plt.show()

model = XGBClassifier(
    # max_depth=10,
    early_stopping_rounds=2,
    n_estimators=2000,
    device="gpu",
    nthread=n_threads,
)

print(f"[TRAINING] ntread: {n_threads} :: x_shape: {X_train.shape}")

model.fit(
    X_train,
    y_train,
    verbose=True,
    eval_set=[(X_train, y_train), (X_test, y_test)],
)

print("[PREDICTING]")

pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=pred))
