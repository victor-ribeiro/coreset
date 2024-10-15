from torchtext import data

print(f"{data=}")
exit()
###################################
import re
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from functools import partial

from coreset.train import train
from coreset.lazzy_greed import lazy_greed
from coreset.utils import random_sampler
from coreset.model.basics import TorchLearner
from coreset.model.neuralnet import MLP

from torch_utils.data import sampling_dataset, BaseDataset


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    # sent = " ".join(sent).lower()
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent


batch_size = 256

####################################
## preprocessing
####################################

# outfile, DATA_HOME, names, tgt_name = load_config()
DATA_HOME = "/Users/victor/Documents/projects/coreset/data/drugs_review/transformed_drugs_review.pickle"

with open(DATA_HOME, "rb") as file:

    features = pickle.load(file)

features, target = features["features"], features["target"]
target = map(lambda x: 1 if x > 5 else 0, target)
target = np.array([*target])

features = map(clean_sent, features)

features = (
    # CountVectorizer(min_df=3, max_features=1800).fit_transform(features).toarray()
    TfidfVectorizer(min_df=0.01, max_df=0.99, max_features=2000)
    .fit_transform(features)
    .toarray()
)
features = PCA(n_components=400).fit_transform(features)

LazyDataset = sampling_dataset(BaseDataset, partial(lazy_greed, metric="codist"))
RandomDataset = sampling_dataset(BaseDataset, random_sampler)

Loader = partial(DataLoader, shuffle=True, batch_size=batch_size, drop_last=False)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=True
)
####################################
## modeling
####################################

# loss_fn = nn.BCELoss
loss_fn = nn.CrossEntropyLoss
lr = 10e-5
epochs = 500

_, nsize = X_train.shape
size = int(len(target) * 0.05)

base_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
dataset = Loader(BaseDataset(X_train, y_train), batch_size=batch_size)
hist = train(base_model, dataset, loss_fn(), Adam, lr, epochs)
pred = base_model(X_test).astype(int)
print(classification_report(y_pred=pred, y_true=y_test))

plt.plot(hist, label="base Model")

lazy_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
dataset = LazyDataset(features=X_train, target=y_train, coreset_size=size)
dataset = Loader(dataset=dataset, batch_size=batch_size)
hist = train(lazy_model, dataset, loss_fn(), Adam, lr, epochs)
pred = lazy_model(X_test).astype(int)
print(classification_report(y_pred=pred, y_true=y_test))

plt.plot(hist, label="lazy_greed")
random_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
dataset = RandomDataset(features=X_train, target=y_train, coreset_size=size)
dataset = Loader(dataset=dataset, batch_size=batch_size)
hist = train(random_model, dataset, loss_fn(), Adam, lr, epochs)
pred = random_model(X_test).astype(int)
print(classification_report(y_pred=pred, y_true=y_test))
plt.plot(hist, label="random")
plt.legend()
plt.show()
