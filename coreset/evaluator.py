from typing import Any
import pandas as pd

from coreset.dataset.transform import pipeline
from coreset.utils import split_dataset

import matplotlib.pyplot as plt

REPEAT = 50

TASKS = {
    "binary_classification": "logloss",
    "mlabel_classification": "mlogloss",
    "regression": "rmmse",
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
        # colocar eval_metric como parametro
        preprocessing = pipeline(
            *self.preprocessing, split_dataset(label=self.lbl_name)
        )
        for _ in range(self.repeat):
            data = self._data
            model = self.model()
            (X_train, y_train), (X_test, y_test) = preprocessing(data)
            n_samples = len(X_train)
            if sampler:
                sset = sampler(X_train)
                X_train = X_train[sset]
                y_train = y_train[sset]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            for metric in self._metrics:
                result = {}
                try:
                    result["sampler"] = (
                        sampler.func.__name__ if sampler else "full_dataset"
                    )
                except:
                    result["sampler"] = sampler.__name__ if sampler else "full_dataset"
                result["sample_size"] = len(X_train)
                result["train_size"] = n_samples
                try:
                    result["metric"] = metric.__name__
                except:
                    result["metric"] = metric.func.__name__
                result["value"] = metric(y_test, pred)
                self.result.append(result)


class TrainCurve(ExperimentTemplate):
    __slots__ = ("hist", "eval_metric")

    def __init__(self, data, model, lbl_name, task, repeat=1, epochs=30) -> None:
        super().__init__(data, model, lbl_name, repeat, epochs)
        self.hist = []
        self.eval_metric = TASKS[task]

    def __call__(self, sampler=None) -> Any:
        preprocessing = pipeline(
            *self.preprocessing, split_dataset(label=self.lbl_name)
        )
        for _ in range(self.repeat):
            data = self._data
            model = self.model(
                sampling_methos="gradient_based",
                n_estimators=30,
                eval_metric=self.eval_metric,
            )
            # model = self.model(n_estimators=30, eval_metric="auc")
            (X_train, y_train), (X_test, y_test) = preprocessing(data)
            n_samples = len(X_train)
            if sampler:
                sset = sampler(X_train)
                X_train = X_train[sset]
                y_train = y_train[sset]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
            )
            pred = model.predict(X_test)

            for metric in self._metrics:
                result = {}
                try:
                    result["sampler"] = (
                        sampler.func.__name__ if sampler else "full_dataset"
                    )
                except:
                    result["sampler"] = sampler.__name__ if sampler else "full_dataset"
                result["sample_size"] = len(X_train)
                result["train_size"] = n_samples
                try:
                    result["metric"] = metric.__name__
                except:
                    result["metric"] = metric.func.__name__
                result["value"] = metric(y_test, pred)
                self.result.append(result)
            hist = {
                f"{result[f'sampler']}_{len(X_train)}": model.evals_result()[
                    "validation_0"
                ][model.eval_metric]
            }
            self.hist.append(hist)

    @property
    def history(self):
        hist = pd.DataFrame.from_records(self.hist).T
        hist.columns = range(1, self.epochs)
        hist["sampler"] = hist.index
        hist.index = range(len(hist))

        hist["sampler"], hist["sample_size"] = (
            hist["sampler"].str.extract("([a-zA-Z_]+)?([0-9]+)").values.T
        )
        hist["sampler"] = hist["sampler"].str.replace("_$", "", regex=True)
        return hist
