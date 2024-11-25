import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Activation,
    Dense,
    Conv1D,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPool2D,
    MaxPool1D,
    Dropout,
)
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping


from coreset.lazzy_greed import freddy
from coreset.utils import craig_baseline
from coreset.utils import random_sampler
from coreset.environ import load_config


from time import time as timer

# from timeit import default_timer as timer


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []
        # self.starttime = timer()

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        elapsed = timer() - self.starttime
        self.logs.append(elapsed)


outfile, DATA_HOME, names, tgt_name = load_config()
result = []

##########################################################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n, r, c = X_train.shape
c = r * c
# reg, epochs, batch_size, core_size = 10e-4, 3000, 128, 0.1
# reg, epochs, batch_size, core_size = 10e-4, 15, 256 * 2, 0.4
reg, epochs, batch_size, core_size = 10e-4, 6000, 256, 0.1

# X_train, X_test = X_train.reshape((n, c)), X_test.reshape((len(X_test), c))

X_train = np.vstack([X_train, X_test])
y_train = np.hstack([y_train, y_test])
y_train = to_categorical(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)
n, *size = X_train.shape
for _ in range(1):
    # for _ in range(1):
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    cb = TimingCallback()
    model = Sequential()
    model.add(Input(size, name="image"))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=100, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
    model.add(Activation("sigmoid"))
    model.add(Dense(10, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))

    model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
    hist_ = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[cb],
    )
    hist_ = hist_.history
    tmp = pd.DataFrame(hist_)
    tmp["sampler"] = "Full dataset"
    tmp["elapsed"] = np.cumsum(cb.logs).round()
    tmp["epoch"] = np.arange(epochs)
    result.append(tmp)

    del hist_
    del tmp
    del model
    del cb

    #     ##########################################################################################
    #     ##########################################################################################
    #     ##########################################################################################

    cb = TimingCallback()

    ft = PCA(n_components=10).fit_transform(X_train.reshape((n, c)))
    idx = freddy(ft, K=int(len(X_train) * core_size), batch_size=256, gamma=0.5)
    X_lazy = X_train[idx]
    y_lazy = y_train[idx]
    model = Sequential()
    model.add(Input(size, name="image"))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=100, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
    model.add(Activation("sigmoid"))
    model.add(Dense(10, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))

    model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
    hist_ = model.fit(
        X_lazy,
        y_lazy,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[cb],
    )
    hist_ = hist_.history

    tmp = pd.DataFrame(hist_)
    tmp["sampler"] = "freddy"
    tmp["elapsed"] = np.cumsum(cb.logs).round()
    tmp["epoch"] = np.arange(epochs)
    result.append(tmp)

    del hist_
    del tmp
    del model
    del cb

    #     ##########################################################################################
    #     ##########################################################################################
    #     ##########################################################################################

    cb = TimingCallback()
    ft = PCA(n_components=10).fit_transform(X_train.reshape((n, c)))
    idx = craig_baseline(ft, K=int(len(X_train) * core_size))
    X_craig = X_train[idx]
    y_craig = y_train[idx]
    model = Sequential()
    model.add(Input(size, name="image"))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=100, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
    model.add(Activation("sigmoid"))
    model.add(Dense(10, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))
    model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
    hist_ = model.fit(X_craig, y_craig, batch_size=batch_size, epochs=epochs)
    hist_ = hist_.history
    hist_ = model.fit(
        X_craig,
        y_craig,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[cb],
    )
    hist_ = hist_.history

    tmp = pd.DataFrame(hist_)
    tmp["sampler"] = "CRAIG"
    tmp["elapsed"] = np.cumsum(cb.logs).round()
    tmp["epoch"] = np.arange(epochs)
    result.append(tmp)

    del hist_
    del tmp
    del model
    del cb

    #     ##########################################################################################
    #     ##########################################################################################
    #     ##########################################################################################

    # cb = TimingCallback()

    # idx = random_sampler(X_train, K=int(len(X_train) * core_size))
    # X_random = X_train[idx]
    # y_random = y_train[idx]
    # model = Sequential()
    # model.add(Input(size, name="image"))
    # model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    # model.add(MaxPool1D())
    # model.add(Conv1D(filters=100, kernel_size=3, activation="relu"))
    # model.add(MaxPool1D())
    # model.add(Dropout(0.1))
    # model.add(Flatten())
    # model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
    # model.add(Activation("sigmoid"))
    # model.add(Dense(10, kernel_regularizer=l2(reg)))
    # model.add(Activation("softmax"))

    # model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")

    # hist_ = model.fit(
    #     X_random,
    #     y_random,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     validation_data=(X_test, y_test),
    #     callbacks=[cb],
    # )
    # hist_ = hist_.history

    # tmp = pd.DataFrame(hist_)
    # tmp["sampler"] = "Random sampler"
    # tmp["elapsed"] = np.cumsum(cb.logs).round()
    # tmp["epoch"] = np.arange(epochs)
    # result.append(tmp)

    # del hist_
    # del tmp
    # del model
    # del cb

#     ##########################################################################################
#     ##########################################################################################

result = pd.concat(result, ignore_index=True)
result.to_csv(outfile)

import seaborn as sns

fig, ax = plt.subplots(1, 2)

sns.lineplot(
    data=result,
    x="elapsed",
    y="val_accuracy",
    hue="sampler",
    style="sampler",
    ax=ax[0],
    # errorbar="sd",
    errorbar=("sd", 0.5),
    # err_style="bars",
    alpha=0.5,
)
sns.lineplot(
    data=result,
    x="elapsed",
    y="loss",
    hue="sampler",
    style="sampler",
    ax=ax[1],
    # errorbar="sd",
    errorbar=("sd", 0.5),
    # err_style="bars",
    alpha=0.5,
)
plt.show()
