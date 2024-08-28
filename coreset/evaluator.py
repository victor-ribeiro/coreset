from typing import Any
import pandas as pd

from coreset.dataset.transform import pipeline
from coreset.utils import split_dataset

REPEAT = 50


class BaseExperiment:
    __slots__ = (
        "_data",
        "model",
        "lbl_name",
        "preprocessing",
        "metrics",
        "result",
        "repeat",
    )

    def __init__(self, data, model, lbl_name, repeat=1, name="") -> None:
        self._data = data
        self.model = model
        self.lbl_name = lbl_name
        self.repeat = repeat
        self.preprocessing = []
        self.metrics = []
        self.result = []

    def __call__(self, sampler=None) -> Any:
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

            for metric in self.metrics:
                result = {}
                try:
                    result["sampler"] = sampler.func.__name__ if sampler else None
                except:
                    result["sampler"] = sampler.__name__ if sampler else None
                result["sample_size"] = len(X_train)
                result["sample_prop"] = len(X_train) / n_samples
                try:
                    result["metric"] = metric.__name__
                except:
                    result["metric"] = metric.func.__name__
                result["value"] = metric(y_test, pred)
                self.result.append(result)
        return pd.DataFrame.from_records(self.result)

    def __mul__(self, other):
        return self.__call__(other)

    def register_metrics(self, *f_):
        for i in f_:
            self.metrics.append(i)

    def register_preprocessing(self, *f_):
        for i in f_:
            self.preprocessing.append(i)
