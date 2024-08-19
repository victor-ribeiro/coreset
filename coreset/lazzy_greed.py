###############################################
### |S| <= 1 + log max F( e | [] ) x | S* | ###
###############################################

import heapq
import math
import numpy as np
from functools import lru_cache, partial, reduce
from itertools import batched
from datetime import datetime

from coreset.metrics import METRICS
from coreset.dataset.dataset import Dataset


class Queue(list):
    def __init__(self, *iterable):
        super().__init__(*iterable)
        heapq._heapify_max(self)

    def append(self, item: "Any"):
        super().append(item)
        heapq._siftdown_max(self, 0, len(self) - 1)

    def pop(self, index=-1):
        el = super().pop(index)
        if not self:
            return el
        val, self[0] = self[0], el
        heapq._siftup_max(self, 0)
        return val

    @property
    def head(self):
        return self.pop()

    def push(self, idx, score):
        item = (idx, score)
        self.append(item)


def base_inc(alpha=1):
    # mudar isso aqui
    return math.log(1 + alpha)


def utility_score(e, sset, /, norm_fn, alpha=1):
    norm = 1 / base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1)
    return norm * math.log(1 + argmax.sum() * f_norm)


def lazy_greed(
    dataset: Dataset,
    base_inc,
    alpha=1,
    metric="similarity",
    K=1,
    batch_size=32,
):
    # basic config
    base_inc = base_inc(alpha)
    argmax = np.zeros(dataset.size)
    score = 0
    metric = METRICS[metric]
    q = Queue()
    sset = []
    vals = []

    for i, (D, V) in enumerate(
        zip(
            metric(dataset, batch_size=batch_size),
            batched(dataset.index, batch_size),
        )
    ):
        size = len(D)
        [q.push(base_inc, idx) for idx in zip(V, range(size))]

        # if len(sset) >= K:
        #     break
        while q and len(sset) < K:
            _, idx_s = q.head
            s = D[idx_s[1]]
            score_s = utility_score(s, argmax, alpha)  # F( e | S )
            inc = score_s - score
            if inc < 0:
                continue
            if not q:
                break
            score_t, idx_t = q.head
            if inc > score_t:
                argmax = np.maximum(argmax, s)
                score = utility_score(s, argmax, alpha)
                sset.append(idx_s[0])
                vals.append(score)
                q.push(score_t, idx_t)
            q.push(inc, idx_s)
        else:
            argmax = np.maximum(argmax, s)
            score = utility_score(s, argmax, alpha)
            sset.append(idx_s[0])
            vals.append(score)
    return sset, vals


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report

    ft, lbl = make_classification(
        n_samples=10_000,
        n_features=20,
        n_informative=10,
        n_classes=10,
    )
    data = pd.DataFrame(ft)
    data["label"] = lbl
    train, test = train_test_split(data, test_size=0.1)
    ytrain = train.pop("label")
    ytest = test.pop("label")

    model = HistGradientBoostingClassifier()
    start = datetime.now().timestamp()
    model.fit(train, ytrain)
    print(classification_report(ytest, model.predict(test)))
    end = datetime.now().timestamp()

    print(end - start)

    dataset = Dataset(train)
    start = datetime.now().timestamp()
    sset, vals = lazy_greed(dataset, base_inc, alpha=2, K=100, batch_size=1024)
    dataset = dataset[sset]

    outro_model = HistGradientBoostingClassifier()
    outro_model.fit(dataset, ytrain.iloc[sset])
    print(classification_report(ytest, outro_model.predict(test)))
    end = datetime.now().timestamp()
    print(end - start)
