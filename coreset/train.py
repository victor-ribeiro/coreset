from functools import singledispatch
import numpy as np
from torch.optim import Adam
from torch_utils.train import train_loop, eval_train
from .model.basics import SklearnLearner, TorchLearner
from time import time


@singledispatch
def train(
    learner: TorchLearner,
    data_train,
    loss_fn,
    optmizer,
    lr,
    epochs,
    data_test=None,
    data_valid=None,
):
    model = learner._model
    optim = optmizer(model.parameters(), lr=lr)
    # fazer o treinamento, retornar o modelo
    hist = []
    elapsed = []
    t = time()
    for loss in train_loop(data_train, loss_fn, optim, model, epochs):
        hist.append(loss)
        elapsed.append(time() - t)
    learner._model = model
    learner.fited = True
    return hist, elapsed


@train.register(SklearnLearner)
def _(learner: SklearnLearner, data_train):
    X, y = data_train
    learner.fit(X, y)
    return learner
