from functools import singledispatch
from torch.optim import Adam
from torch_utils.train import train_loop, eval_train
from .model.basics import SklearnLearner, TorchLearner


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
    optim = optmizer(model.parameters(), lr=lr, momentum=0.09)
    # optim = Adam(learner._model.parameters(), lr=lr)
    if data_test and not data_train:
        train_fn = eval_train(data_test)
    elif data_test and data_train:
        train_fn = eval_train(data_test, data_valid)
    else:
        train_fn = train_loop
    # fazer o treinamento, retornar o modelo
    hist = [_ for _ in train_fn(data_train, loss_fn, optim, model, epochs)]
    learner._model = model
    learner.fited = True
    return hist


@train.register(SklearnLearner)
def _(learner: SklearnLearner, data_train):
    X, y = data_train
    learner.fit(X, y)
    return learner
