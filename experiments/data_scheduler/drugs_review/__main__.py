import re
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial

from coreset.train import train
from coreset.lazzy_greed import freddy
from coreset.model.basics import TorchLearner
from coreset.model.neuralnet import MLP
from coreset.environ import load_config
from coreset.utils import random_sampler, craig_baseline
from torch_utils.data import sampling_dataset, BaseDataset
from coreset.evaluator import REPEAT

outfile, DATA_HOME, names, tgt_name = load_config()


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    # sent = " ".join(sent).lower()
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent


REPEAT = 1

batch_size = 256 * 5

####################################
## preprocessing
####################################

# outfile, DATA_HOME, names, tgt_name = load_config()

with open(DATA_HOME, "rb") as file:

    features = pickle.load(file)

# SGD = partial(SGD, weight_decay=10e-2, momentum=0.9, nesterov=True)

features, target = features["features"], features["target"]
target = map(lambda x: 1 if x > 5 else 0, target)
target = np.array([*target])
target = OneHotEncoder().fit_transform(target.reshape(-1, 1)).toarray()

features = map(clean_sent, features)

features = (
    CountVectorizer(min_df=3, max_features=1800).fit_transform(features).toarray()
)

features = PCA(n_components=300).fit_transform(features)

LazyDataset = sampling_dataset(BaseDataset, freddy)
RandomDataset = sampling_dataset(BaseDataset, random_sampler)
CraigDataset = sampling_dataset(BaseDataset, craig_baseline)

Loader = partial(DataLoader, shuffle=True, batch_size=batch_size, drop_last=False)

result = pd.DataFrame()

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=True
)

nsize, output_size = y_train.shape
model_args = {"input_size": nsize, "output_size": output_size}
# Adam = partial(Adam, weight_decay=10e-4, betas=[0.9, 0.999])
for i in range(REPEAT):
    ####################################
    ## modeling
    ####################################

    loss_fn = nn.BCEWithLogitsLoss

    lr = 10e-3
    epochs = 15

    size = int(len(target) * 0.1)

    # craig_model = TorchLearner(MLP, {"input_size": nsize, "n_layers": 5})
    # dataset = CraigDataset(features=X_train, target=y_train, coreset_size=size)
    # dataset = Loader(dataset=dataset, batch_size=batch_size)
    # hist, elapsed = train(craig_model, dataset, loss_fn(), Adam, lr, epochs)
    # tmp = pd.DataFrame({"hist": hist, "elapsed": elapsed})
    # tmp["method"] = "craig_baseline"
    # result = pd.concat([result, tmp], ignore_index=True)
    # del craig_model
    # del dataset

    base_model = TorchLearner(MLP, **model_args)
    dataset = Loader(BaseDataset(X_train, y_train), batch_size=batch_size)
    hist, elapsed = train(base_model, dataset, loss_fn(), SGD, lr, epochs)
    tmp = pd.DataFrame({"hist": hist, "elapsed": elapsed})
    tmp["method"] = "full_dataset"
    result = pd.concat([result, tmp], ignore_index=True)
    pred = base_model(X_test).astype(int)
    print(classification_report(y_pred=pred, y_true=y_test))
    del base_model
    del dataset

    lazy_model = TorchLearner(MLP, **model_args)
    dataset = LazyDataset(features=X_train, target=y_train, coreset_size=size)
    dataset = Loader(dataset=dataset, batch_size=batch_size)
    hist, elapsed = train(lazy_model, dataset, loss_fn(), SGD, lr, epochs)
    tmp = pd.DataFrame({"hist": hist, "elapsed": elapsed})
    tmp["method"] = "lazy_greed"
    result = pd.concat([result, tmp], ignore_index=True)

    pred = lazy_model(X_test).astype(int)
    print(classification_report(y_pred=pred, y_true=y_test))
    del lazy_model
    del dataset
    # plt.plot(elapsed, hist, label="lazy_greed")

    random_model = TorchLearner(MLP, **model_args)
    dataset = RandomDataset(features=X_train, target=y_train, coreset_size=size)
    dataset = Loader(dataset=dataset, batch_size=batch_size)
    hist, elapsed = train(random_model, dataset, loss_fn(), SGD, lr, epochs)
    tmp = pd.DataFrame({"hist": hist, "elapsed": elapsed})
    tmp["method"] = "random_sampler"
    result = pd.concat([result, tmp], ignore_index=True)
    pred = random_model(X_test).astype(int)
    print(classification_report(y_pred=pred, y_true=y_test))
    del random_model
    del dataset


result.to_csv(outfile, index=False)

# sns.lineplot(data=result, x="elapsed", y="hist", hue="method")
# sns.lineplot(data=result, x='elapsed', y="hist", hue="method")
# sns.lineplot(data=result, x="elapsed", y="hist", hue=result.columns[-1])
# plt.legend()
# plt.show()
