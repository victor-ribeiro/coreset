from typing import Any
import pandas as pd

from torch.utils.data import DataLoader

from freddy.model.basics import TorchLearner
from freddy.dataset.transform import pipeline
from freddy.dataset.utils import split_dataset
from freddy.train import train
from torch_utils.data import sampling_dataset, BaseDataset

import matplotlib.pyplot as plt

# REPEAT = 50
# REPEAT = 10
REPEAT = 5

TASKS = {
    "binary_classification": "logloss",
    "mlabel_classification": "mlogloss",
    # "mlabel_classification": "merror",
    "regression": "rmse",
}


class ExperimentTemplate:
    __slots__ = (
        "_data",
        "model",
        "lbl_name",
        "preprocessing",
        "_metrics",
        "result",
        "repeat",
        "epochs",
    )

    def __init__(self, data, model, lbl_name, repeat=1, epochs=30) -> None:
        self._data = data
        self.model = model
        self.lbl_name = lbl_name
        self.repeat = repeat
        self.preprocessing = []
        self._metrics = []
        self.result = []
        self.epochs = epochs

    def __call__(self, sampler=None) -> Any:
        pass

    @property
    def metrics(self):
        return pd.DataFrame.from_records(self.result)

    def __mul__(self, other):
        return self.__call__(other)

    def register_metrics(self, *f_):
        for i in f_:
            self._metrics.append(i)

    def register_preprocessing(self, *f_):
        for i in f_:
            self.preprocessing.append(i)


class BaseExperiment(ExperimentTemplate):
    def __call__(self, sampler=None) -> Any:
        preprocessing = pipeline(*self.preprocessing)
        split_fn = split_dataset(label=self.lbl_name)
        data = preprocessing(self._data)
        (X_train, y_train), (X_test, y_test) = split_fn(data)
        n_samples = len(X_train)
        for _ in range(self.repeat):
            model = self.model()
            if sampler:
                sset = sampler(X_train)
                _X_train = X_train[sset]
                _y_train = y_train[sset]
            else:
                _X_train = X_train
                _y_train = y_train

            model.fit(_X_train, _y_train)
            pred = model.predict(X_test)

            for metric in self._metrics:
                result = {}
                try:
                    result["sampler"] = (
                        sampler.func.__name__ if sampler else "full_dataset"
                    )
                except Exception as e:
                    result["sampler"] = sampler.__name__ if sampler else "full_dataset"
                result["sample_size"] = len(_X_train)
                result["train_size"] = n_samples
                try:
                    result["metric"] = metric.__name__
                except:
                    result["metric"] = metric.func.__name__
                result["value"] = metric(y_test, pred)
                self.result.append(result)


class BSizeExperiment(ExperimentTemplate):

    def __call__(self, batch_size=None, sampler=None) -> Any:
        preprocessing = pipeline(*self.preprocessing)
        split_fn = split_dataset(label=self.lbl_name)
        data = preprocessing(self._data)
        (X_train, y_train), (X_test, y_test) = split_fn(data)
        n_samples = len(X_train)
        for _ in range(self.repeat):
            model = self.model()
            if sampler:
                if batch_size:
                    sset = sampler(X_train, batch_size=batch_size)
                else:
                    sset = sampler(X_train)
                _X_train = X_train[sset]
                _y_train = y_train[sset]
            else:
                _X_train = X_train
                _y_train = y_train

            model.fit(_X_train, _y_train)
            pred = model.predict(X_test)

            for metric in self._metrics:
                result = {}
                try:
                    result["sampler"] = (
                        sampler.func.__name__ if sampler else "full_dataset"
                    )
                except Exception as e:
                    result["sampler"] = sampler.__name__ if sampler else "full_dataset"
                result["sample_size"] = len(_X_train)
                result["train_size"] = n_samples
                result["batch_size"] = batch_size
                try:
                    result["metric"] = metric.__name__
                except:
                    result["metric"] = metric.func.__name__
                result["value"] = metric(y_test, pred)
                self.result.append(result)


class TrainCurve(ExperimentTemplate):

    # TODO: implementar o método de treinamento
    # TODO: compatibilizar com TorchLearner
    __slots__ = ("hist", "train_fn", "elapsed", "params", "lr")

    def __init__(
        self, data, model, lbl_name, train_fn, repeat=1, epochs=30, lr=10e-4, params={}
    ) -> None:
        super().__init__(data, model, lbl_name, repeat, epochs)
        self.hist = []
        self.elapsed = []
        self.train_fn = train_fn
        self.params = params
        self.lr = lr

    def __call__(
        self, model, loss_fn, optmizer, batch_size=32, sampler=None, **model_args
    ) -> Any:
        preprocessing = pipeline(*self.preprocessing)
        split_fn = split_dataset(label=self.lbl_name)
        data = preprocessing(self._data)
        (X_train, y_train), (X_test, y_test) = split_fn(data)
        for _ in range(self.repeat):

            if sampler:
                dataset = sampling_dataset(BaseDataset, sampler)
            else:
                dataset = BaseDataset

            data_loader = DataLoader(
                dataset(features=X_train, target=y_train),
                shuffle=True,
                batch_size=batch_size,
                drop_last=False,
            )
            hist, elapsed = train(
                model, data_loader, optmizer, loss_fn(), self.lr, self.epochs
            )
            result = {
                "hist": hist,
                "elapsed": elapsed,
            }
            # n_samples = len(X_train)
            # if sampler:
            #     sset = sampler(X_train)
            #     X_train = X_train[sset]
            #     y_train = y_train[sset]

            # model.fit(
            #     X_train,
            #     y_train,
            #     eval_set=[(X_train, y_train), (X_test, y_test)],
            #     verbose=False,
            # )
            # try:
            #     mthd_name = sampler.func.__name__ if sampler else "full_dataset"
            # except:
            #     mthd_name = sampler.__name__ if sampler else "full_dataset"

            # sample_size = len(X_train)
            # train_size = n_samples

            # self.result.append(
            #     [
            #         mthd_name,
            #         "train",
            #         sample_size,
            #         train_size,
            #         *model.evals_result()["validation_0"][model.eval_metric],
            #     ]
            # )
            # self.result.append(
            #     [
            #         mthd_name,
            #         "test",
            #         sample_size,
            #         train_size,
            #         *model.evals_result()["validation_1"][model.eval_metric],
            #     ]
            # )

    @property
    def history(self):
        hist = pd.DataFrame(self.result)
        hist.columns = [
            "sampler",
            "eval",
            "sample_size",
            "train_size",
            *range(1, self.epochs + 1),
        ]
        return hist
