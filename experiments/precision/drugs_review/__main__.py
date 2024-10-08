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

import re


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent
    return sent.split()


def build_vocab(dataset):
    vocab = " ".join(dataset)
    vocab = clean_sent(vocab)
    vocab = set(vocab)
    return {w: i for w, i in enumerate(vocab)}


import pickle

from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    CountVectorizer,
)

import torch
from torch_utils.data import BaseDataset, sampling_dataset
from torch_utils.train import train_loop

from coreset.train import train
from coreset.model.neuralnet import MLP
from coreset.model.basics import TorchLearner
from coreset.lazzy_greed import lazy_greed
from coreset.utils import random_sampler
from coreset.kmeans import kmeans_sampler
from sklearn.utils import class_weight


batch_size = 256
# loss_fn = nn.CrossEntropyLoss
lr = 10e-4
epochs = 30


lazy_greed = partial(lazy_greed, batch_size=512)
LazyDataset = sampling_dataset(BaseDataset, lazy_greed)
RandomDataset = sampling_dataset(BaseDataset, random_sampler)
KMeansDataset = sampling_dataset(BaseDataset, kmeans_sampler)
encoder = CountVectorizer(min_df=5, max_features=2000)


with open(DATA_HOME, "rb") as file:

    dataset = pickle.load(file)

Loader = partial(DataLoader, shuffle=True, batch_size=batch_size, drop_last=False)
import numpy as np
from torch import functional as F
from torch import nn
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA, TruncatedSVD

features, target = dataset.values()
target = np.array(target)

classes = np.unique(target)
W = np.ones(len(target))


for i, w in enumerate(
    class_weight.compute_class_weight("balanced", classes=classes, y=target)
):
    idx = np.where(target == i)
    W[idx] = W[idx] * w
W = torch.tensor(W)

loss_fn = partial(nn.BCELoss, weight=W)

features = map(clean_sent, features)
features = encoder.fit_transform(features).toarray()
features = FastICA(n_components=256).fit_transform(features)
#
target = [*map(lambda x: 1 if x > 5 else 0, target)]


target = OneHotEncoder().fit_transform(np.reshape(target, (-1, 1))).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=True
)

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

_, nsize = X_train.shape
model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})

dataset = BaseDataset(features=X_train, target=y_train)
dataset = Loader(dataset=dataset)
hist = train(model, dataset, loss_fn(), Adam, lr, epochs)

pred = model(X_test)
print(
    classification_report(
        y_pred=np.argmax(pred, axis=1), y_true=np.argmax(y_test, axis=1)
    )
)

plt.plot(hist, label="full dataset")

lazy_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
size = int(len(target) * 0.05)
dataset = LazyDataset(features=X_train, target=y_train, coreset_size=size)
dataset = Loader(dataset=dataset)
hist = train(lazy_model, dataset, loss_fn(), Adam, lr, epochs)
pred = lazy_model(X_test).astype(int)
print(
    classification_report(
        y_pred=np.argmax(pred, axis=1), y_true=np.argmax(y_test, axis=1)
    )
)
plt.plot(hist, label="lazy_greed")

random_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
dataset = RandomDataset(features=X_train, target=y_train, coreset_size=size)
dataset = Loader(dataset=dataset)
hist = train(random_model, dataset, loss_fn(), Adam, lr, epochs)
pred = random_model(X_test).astype(int)
print(
    classification_report(
        y_pred=np.argmax(pred, axis=1), y_true=np.argmax(y_test, axis=1)
    )
)
plt.plot(hist, label="random")

k_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
dataset = KMeansDataset(features=X_train, target=y_train, coreset_size=size)
dataset = Loader(dataset=dataset)
hist = train(k_model, dataset, loss_fn(), Adam, lr, epochs)
pred = model(X_test).astype(int)
print(classification_report(y_pred=pred, y_true=y_test))
plt.plot(hist, label="kmeans")
plt.legend()
plt.show()
