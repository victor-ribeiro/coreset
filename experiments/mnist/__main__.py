import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.callbacks import Callback

# from keras import backend as K

# K.tensorflow_backend._get_available_gpus()

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


##########################################################################################
gpus = tf.config.experimental.list_physical_devices("GPU")

##########################################################################################

outfile, DATA_HOME, names, tgt_name = load_config()
result = []

##########################################################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n, r, c = X_train.shape
c = r * c
reg, epochs, batch_size, core_size = 10e-4, 15, 256 * 4, 0.4

X_train, X_test = X_train.reshape((n, c)), X_test.reshape((len(X_test), c))

X_train = np.vstack([X_train, X_train])
y_train = np.hstack([y_train, y_train])
y_train = to_categorical(y_train)

for gpu in gpus:
    with tf.device(gpu.name):
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=True
        )

        for _ in range(15):
            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            cb = TimingCallback()

            model = Sequential()
            model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
            model.add(Activation("sigmoid"))
            model.add(Dense(10, kernel_regularizer=l2(reg)))
            model.add(Activation("softmax"))

            model.compile(
                loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd"
            )
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

            result.append(tmp)

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            from coreset.lazzy_greed import fastcore

            cb = TimingCallback()
            alpha = 0.1
            beta = 1.1

            idx = fastcore(
                X_train,
                K=int(len(X_train) * core_size),
                batch_size=256 * 6,
                # batch_size=batch_size,
                alpha=alpha,
                beta=beta,
            )
            X_lazy = X_train[idx]
            y_lazy = y_train[idx]
            model = Sequential()
            model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
            model.add(Activation("sigmoid"))
            model.add(Dense(10, kernel_regularizer=l2(reg)))
            model.add(Activation("softmax"))
            model.compile(
                loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd"
            )
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
            tmp["sampler"] = "FastCORE"
            tmp["elapsed"] = np.cumsum(cb.logs).round()

            result.append(tmp)

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            from coreset.utils import craig_baseline
            from sklearn.decomposition import PCA

            cb = TimingCallback()
            ft = PCA(n_components=10).fit_transform(X_train)
            idx = craig_baseline(ft, K=int(len(X_train) * core_size))
            X_random = X_train[idx]
            y_random = y_train[idx]
            model = Sequential()
            model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
            model.add(Activation("sigmoid"))
            model.add(Dense(10, kernel_regularizer=l2(reg)))
            model.add(Activation("softmax"))

            model.compile(
                loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd"
            )
            hist_ = model.fit(X_random, y_random, batch_size=batch_size, epochs=epochs)
            hist_ = hist_.history
            hist_ = model.fit(
                X_random,
                y_random,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[cb],
            )
            hist_ = hist_.history

            tmp = pd.DataFrame(hist_)
            tmp["sampler"] = "CRAIG"
            tmp["elapsed"] = np.cumsum(cb.logs).round()
            result.append(tmp)

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            from coreset.utils import random_sampler

            cb = TimingCallback()

            idx = random_sampler(X_train, K=int(len(X_train) * core_size))
            X_random = X_train[idx]
            y_random = y_train[idx]
            model = Sequential()
            model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
            model.add(Activation("sigmoid"))
            model.add(Dense(10, kernel_regularizer=l2(reg)))
            model.add(Activation("softmax"))

            model.compile(
                loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd"
            )

            hist_ = model.fit(
                X_random,
                y_random,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=[cb],
            )
            hist_ = hist_.history

            tmp = pd.DataFrame(hist_)
            tmp["sampler"] = "Random sampler"
            tmp["elapsed"] = np.cumsum(cb.logs).round()
            result.append(tmp)

            ##########################################################################################
            ##########################################################################################
with tf.device("/CPU:0"):
    result = pd.concat(result, ignore_index=True)
    result.to_csv(outfile)
# result = pd.concat(result, ignore_index=True)
# result.to_csv(outfile)

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
