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

import nltk

nltk.download("stopwords")

import pandas as pd
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import (
    RegexpTokenizer,
    word_tokenize,
    wordpunct_tokenize,
    SyllableTokenizer,
)

from nltk.stem import PorterStemmer, RSLPStemmer
from nltk.stem.snowball import SnowballStemmer

# from nltk.sem import

# from sklearn.feature_extraction.text import HashingVectorizer as vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize, minmax_scale, quantile_transform

from xgboost import XGBClassifier, XGBRanker
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

data = pd.read_csv(DATA_HOME, sep="\t", index_col=0)
data.dropna(axis="index", inplace=True)
data.date = pd.to_datetime(data.date)
tgt = data.pop("rating").values.astype(int).reshape(-1, 1) - 1

tokenizer = RegexpTokenizer(r"\w+")

stop_words = list(stopwords.words("english"))
stop_words += ["i"]


for name in data["drugName"].unique():
    stop_words += [name.lower()]
for name in data["condition"].unique():
    stop_words += [name.lower()]

# 'drugName', 'condition', 'date', 'usefulCount'
# review = (data.pop("review") + data.pop("condition")).values
data["date"] = OrdinalEncoder().fit_transform(data["date"].values.reshape(-1, 1))


txt2vec = partial(
    vectorizer,
    strip_accents="ascii",
    binary=True,
    tokenizer=wordpunct_tokenize,
    lowercase=True,
)

review = data.pop("review").str.lower().values
review = map(wordpunct_tokenize, review)
review = map(
    lambda tkns: filter(
        lambda x: not x in stop_words and x.isalnum() and not x.isdigit(), tkns
    ),
    review,
)
review = map(lambda x: map(SnowballStemmer("porter").stem, x), review)
review = map(list, review)
review = map(lambda x: "".join(x).lower(), review)
# review = txt2vec(n_features=30).fit_transform(review).toarray()
review = vectorizer(strip_accents="ascii").fit_transform(review)

condition = data["condition"].values
condition = txt2vec(n_features=5).fit_transform(condition).toarray()

dname = data["drugName"].values
# dname = txt2vec(n_features=5).fit_transform(dname).toarray()
dname = ()


# ds = np.vstack(
#     # (*review.T, *dname.T, *condition.T, data["usefulCount"].values, data["date"].values)
#     (*review.T, data["usefulCount"].values, data["date"].values)
# ).T

# ds = quantile_transform(ds, output_distribution="normal")
# ds = normalize(ds)
ds = review


# plt.scatter(*PCA(n_components=2).fit_transform(ds).T, c=tgt)
# plt.show()
# exit()
X_train, X_test, y_train, y_test = train_test_split(ds, tgt, test_size=0.2)

model = XGBClassifier(
    max_depth=100,
    # subsample=0.8,
    # eta=0.02,
    # early_stopping_rounds=10,
    early_stopping_rounds=2,
    n_estimators=2000,
    nthread=4,
)

model.fit(
    X_train,
    y_train,
    verbose=True,
    eval_set=[(X_train, y_train), (X_test, y_test)],
)

pred = model.predict(X_test)
print(classification_report(y_pred=pred, y_true=y_test))
