# # import pandas as pd
# # from functools import partial
# # from xgboost import XGBClassifier

# # from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# # from sklearn.metrics import precision_score, recall_score, f1_score

from coreset.environ import load_config

# # from coreset.utils import random_sampler, hash_encoding, transform_fn, craig_baseline
# # from coreset.lazzy_greed import lazy_greed
# # from coreset.kmeans import kmeans_sampler
# # from coreset.evaluator import BaseExperiment, TrainCurve, REPEAT

# # import matplotlib.pyplot as plt

outfile, DATA_HOME, names, tgt_name = load_config()
# # dataset = pd.read_csv(DATA_HOME, engine="pyarrow", names=names)
# # max_size = len(dataset) * 0.8
# # *names, tgt_name = names


# # def encoding(dataset, *columns):
# #     data = dataset.copy()
# #     for col in columns:
# #         data[col] = OrdinalEncoder().fit_transform(dataset[col].values.reshape(-1, 1))
# #     return data


# # dataset["children"] = dataset["children"].map({"1": 1, "2": 2, "3": 3, "more": 4})
# # print(dataset.shape)
# # dataset[tgt_name] = dataset[tgt_name].map(
# #     lambda x: "recommend" if x == "very_recom" or x == "priority" else x
# # )
# # dataset[tgt_name] = LabelEncoder().fit_transform(dataset[tgt_name]).astype(int)

# # if __name__ == "__main__":
# #     # sampling strategies
# #     smpln = [
# #         partial(lazy_greed, K=int(max_size * 0.01), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.02), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.03), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.04), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.05), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.10), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.15), batch_size=256),
# #         partial(lazy_greed, K=int(max_size * 0.25), batch_size=256),
# #         kmeans_sampler(K=int(max_size * 0.01)),
# #         kmeans_sampler(K=int(max_size * 0.02)),
# #         kmeans_sampler(K=int(max_size * 0.03)),
# #         kmeans_sampler(K=int(max_size * 0.04)),
# #         kmeans_sampler(K=int(max_size * 0.05)),
# #         kmeans_sampler(K=int(max_size * 0.10)),
# #         kmeans_sampler(K=int(max_size * 0.15)),
# #         kmeans_sampler(K=int(max_size * 0.25)),
# #         random_sampler(n_samples=int(max_size * 0.01)),
# #         random_sampler(n_samples=int(max_size * 0.02)),
# #         random_sampler(n_samples=int(max_size * 0.03)),
# #         random_sampler(n_samples=int(max_size * 0.04)),
# #         random_sampler(n_samples=int(max_size * 0.05)),
# #         random_sampler(n_samples=int(max_size * 0.10)),
# #         random_sampler(n_samples=int(max_size * 0.15)),
# #         random_sampler(n_samples=int(max_size * 0.25)),
# #         craig_baseline(0.01),
# #         craig_baseline(0.02),
# #         craig_baseline(0.03),
# #         craig_baseline(0.04),
# #         craig_baseline(0.05),
# #         craig_baseline(0.10),
# #         craig_baseline(0.15),
# #         craig_baseline(0.25),
# #     ]
# #     nursery = BaseExperiment(
# #         dataset,
# #         model=partial(
# #             XGBClassifier,
# #             enable_categorical=True,
# #             grow_policy="lossguide",
# #             n_estimators=30,
# #         ),
# #         lbl_name=tgt_name,
# #         repeat=REPEAT,
# #     )

# #     nursery.register_preprocessing(
# #         hash_encoding("parents", "has_nurs", "form", n_features=5),
# #         transform_fn(encoding, tgt_name, *names[4:]),
# #     )

# #     nursery.register_metrics(
# #         partial(precision_score, average="macro"),
# #     )

# #     nursery()
# #     for sampler in smpln:
# #         nursery(sampler=sampler)
# #     result = nursery.metrics  # base de comparação
# #     result.to_csv(outfile, index=False)

# ##### TODO
# ## [ ] vetorizar
# ## [ ] treinar modelo
# ## [ ] template avaliação de precisão e curva de aprendizado
# ## [ ] _make_dataset.py_ para a pasta data
# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier


# import multiprocessing

# with open(DATA_HOME, "rb") as file:
#     data = pickle.load(file)

# n_threads = int(multiprocessing.cpu_count() / 2)
# min_df = 0.01
# max_df = 1 - min_df

# X_train, y_train = data["features"], data["target"]
# X_train = TfidfVectorizer(
#     max_df=max_df,
#     min_df=min_df,
#     stop_words="english",
#     # max_features=500,
# ).fit_transform(X_train)

# y_train = np.array(y_train)
# y_train = LabelEncoder().fit_transform(y_train.reshape(-1, 1))

# # X_train = CountVectorizer(
# #     max_df=max_df, min_df=min_df, stop_words="english", max_features=200
# # ).fit_transform(X_train)

# # X_train = map(word_tokenize, X_train)
# # X_train = (
# # FeatureHasher(n_features=120, input_type="string").transform(X_train)
# # HashingVectorizer(n_features=30, norm="l1").fit_transform(X_train)
# # )

# X_train, X_test, y_train, y_test = train_test_split(
#     X_train, y_train, test_size=0.2, stratify=y_train
# )

# y_test = np.array(y_test)

# # import matplotlib.pyplot as plt
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import normalize, quantile_transform, minmax_scale

# # x, y = PCA(n_components=2).fit_transform(X_train).T
# # plt.scatter(x, y, c=y_train)
# # plt.title(X_train.shape)
# # plt.show()

# model = XGBClassifier(
#     early_stopping_rounds=2,
#     n_estimators=1000,
#     device="gpu",
#     nthread=n_threads,
# )

# print(f"[TRAINING] ntread: {n_threads} :: x_shape: {X_train.shape}")

# model.fit(
#     X_train,
#     y_train,
#     verbose=True,
#     eval_set=[(X_train, y_train), (X_test, y_test)],
# )

# print("[PREDICTING]")

# pred = model.predict(X_test)

# print(classification_report(y_true=y_test, y_pred=pred))

############### TORCH TEST ###############


import pickle

# from functools import partial
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# from torch import nn
# from torch.optim import Adam, SGD
# from torch.utils.data import DataLoader
# from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


# from torch_utils.data import BaseDataset, sampling_dataset
# from torch_utils.train import train_loop

# from coreset.train import train
# from coreset.model.neuralnet import MLP
# from coreset.model.basics import TorchLearner
# from coreset.lazzy_greed import lazy_greed
# from coreset.utils import random_sampler
# from coreset.kmeans import kmeans_sampler

# batch_size = 256
# # loss_fn = nn.BCELoss
# loss_fn = nn.CrossEntropyLoss
# # loss_fn = nn.NLLLoss
# lr = 10e-3
# epochs = 100


# lazy_greed = partial(lazy_greed, batch_size=512)
# LazyDataset = sampling_dataset(BaseDataset, lazy_greed)
# RandomDataset = sampling_dataset(BaseDataset, random_sampler)
# KMeansDataset = sampling_dataset(BaseDataset, kmeans_sampler)
# encoder = HashingVectorizer(n_features=35)
# encoder = TfidfVectorizer(min_df=0.05)
# encoder = TfidfVectorizer(max_features=100)


with open(DATA_HOME, "rb") as file:

    dataset = pickle.load(file)

# Loader = partial(DataLoader, shuffle=True, batch_size=batch_size, drop_last=False)
# import numpy as np
# from torch import functional as F
# from torch import nn
# import torch

# features, target = dataset.values()

# features = encoder.fit_transform(features).toarray()
# #
# # target = [*map(lambda x: 1 if x > 5 else 0, target)]


# target = OneHotEncoder().fit_transform(np.reshape(target, (-1, 1))).toarray()

# X_train, X_test, y_train, y_test = train_test_split(
#     features, target, test_size=0.2, shuffle=True
# )
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt

# _, nsize = X_train.shape
# model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
# # model = MLP(input_size=nsize)

# size = int(len(target) * 0.1)
# # dataset = LazyDataset(features=X_train, target=y_train, coreset_size=size)
# dataset = BaseDataset(features=X_train, target=y_train)
# dataset = Loader(dataset=dataset)
# hist = train(model, dataset, loss_fn(), SGD, lr, epochs)
# pred = model(X_test).astype(int)
# print(classification_report(y_pred=pred, y_true=y_test))
# plt.plot(hist, label="full dataset")

# model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
# size = int(len(target) * 0.1)
# dataset = LazyDataset(features=X_train, target=y_train, coreset_size=size)
# dataset = Loader(dataset=dataset)
# hist = train(model, dataset, loss_fn(), SGD, lr, epochs)
# pred = model(X_test).astype(int)
# print(classification_report(y_pred=pred, y_true=y_test))
# plt.plot(hist, label="lazy_greed")

# model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
# size = int(len(target) * 0.1)
# dataset = RandomDataset(features=X_train, target=y_train, coreset_size=size)
# dataset = Loader(dataset=dataset)
# hist = train(model, dataset, loss_fn(), SGD, lr, epochs)
# pred = model(X_test).astype(int)
# print(classification_report(y_pred=pred, y_true=y_test))
# plt.plot(hist, label="random")

# model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
# size = int(len(target) * 0.1)
# dataset = KMeansDataset(features=X_train, target=y_train, coreset_size=size)
# dataset = Loader(dataset=dataset)
# hist = train(model, dataset, loss_fn(), SGD, lr, epochs)
# pred = model(X_test).astype(int)
# print(classification_report(y_pred=pred, y_true=y_test))
# plt.plot(hist, label="kmeans")
# plt.legend()
# plt.show()


############### XGBOOST TEST ###############
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    FeatureHasher,
)
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


import multiprocessing
import re


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = filter(lambda x: len(x) > 2, sent)
    sent = filter(lambda x: x.isalnum() and not x.isdigit(), sent)
    return " ".join(sent)
    return sent


with open(DATA_HOME, "rb") as file:
    data = pickle.load(file)

n_threads = int(multiprocessing.cpu_count() / 2)
min_df = 0.01
max_df = 1 - min_df

X_train, y_train = data["features"], data["target"]
# y_train = [*map(lambda x: 1 if x > 5 else 0, y_train)]
X_train = map(clean_sent, X_train)

# X_train = TfidfVectorizer(
#     max_df=max_df, min_df=min_df, stop_words="english"
# ).fit_transform(X_train)

from sklearn.preprocessing import normalize, quantile_transform, minmax_scale
from sklearn.utils import class_weight

# testar com dim = {9,10,11,12}
# X_train = FeatureHasher(
#     n_features=12,
#     input_type="string",
#     # alternate_sign=False,
# ).fit_transform(X_train)

X_train = CountVectorizer(
    # min_df=2, max_df=0.9, max_features=1000
    min_df=5,
    max_df=0.9,
    max_features=1000,
).fit_transform(X_train)

X_train = quantile_transform(X_train.toarray())


# y_train = LabelEncoder().fit_transform(y_train.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train
)

y_test = np.array(y_test)

y_train = np.array(y_train)
classes = np.unique(y_train)

W = np.ones(len(y_train))

for i, w in enumerate(
    class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
):
    idx = np.where(y_train == i)
    W[idx] = W[idx] * w


# print(X_train[0])


# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# x, y = PCA(n_components=2).fit_transform(X_train).T
# plt.scatter(x, y, c=y_train)
# plt.title(X_train.shape)
# plt.show()

# exit()
# (*) -> usados juntos
model = XGBClassifier(
    max_depth=12,
    eta=0.1,
    max_bin=32,  # (*)
    tree_method="hist",  # (*)
    # grow_policy="lossguide",
    early_stopping_rounds=5,
    n_estimators=5000,
    nthread=n_threads,
    max_delta_step=10,
)

print(f"[TRAINING] ntread: {n_threads} :: x_shape: {X_train.shape}")

model.fit(
    X_train,
    y_train,
    verbose=True,
    sample_weight=W,
    eval_set=[(X_train, y_train), (X_test, y_test)],
)

print("[PREDICTING]")

pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=pred))
