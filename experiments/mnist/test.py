import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy


from keras.regularizers import l2

##########################################################################################
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n, r, c = X_train.shape
c = r * c
reg, epochs, batch_size = 10e-4, 50, 256 * 5


# X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train.reshape((n, c)), X_test.reshape((len(X_test), c))

X_train = np.vstack([X_train, X_train])
y_train = np.hstack([y_train, y_train])
y_train = to_categorical(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)

fig, ax = plt.subplots(1, 2)

##########################################################################################
##########################################################################################
##########################################################################################

model = Sequential()
model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
model.add(Activation("sigmoid"))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation("softmax"))

model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
hist_ = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
hist_ = hist_.history
ax[0].plot(hist_["accuracy"], label="Full dataset")
ax[1].plot(hist_["loss"], label="Full dataset")


##########################################################################################
##########################################################################################
##########################################################################################
from coreset.lazzy_greed import lazy_greed

alpha = 0.999
beta = 1 - alpha

idx = lazy_greed(
    X_train, K=int(len(X_train) * 0.3), batch_size=256 * 4, beta=beta, alpha=alpha
)
X_lazy = X_train[idx]
y_lazy = y_train[idx]
model = Sequential()
model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
model.add(Activation("sigmoid"))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation("softmax"))

model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
hist_ = model.fit(X_lazy, y_lazy, batch_size=batch_size, epochs=epochs)
hist_ = hist_.history
ax[0].plot(hist_["accuracy"], label="FastCORE")
ax[1].plot(hist_["loss"], label="FastCORE")

##########################################################################################
##########################################################################################
##########################################################################################
from coreset.utils import craig_baseline

idx = craig_baseline(X_train, K=int(len(X_train) * 0.3))
X_random = X_train[idx]
y_random = y_train[idx]
model = Sequential()
model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
model.add(Activation("sigmoid"))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation("softmax"))

model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
hist_ = model.fit(X_random, y_random, batch_size=batch_size, epochs=epochs)
hist_ = hist_.history
ax[0].plot(hist_["accuracy"], label="CRAIG")
ax[1].plot(hist_["loss"], label="CRAIG")

##########################################################################################
##########################################################################################
##########################################################################################
from coreset.utils import random_sampler

idx = random_sampler(X_train, K=int(len(X_train) * 0.3))
X_random = X_train[idx]
y_random = y_train[idx]
model = Sequential()
model.add(Dense(100, input_dim=c, kernel_regularizer=l2(reg)))
model.add(Activation("sigmoid"))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation("softmax"))

model.compile(loss=CategoricalCrossentropy(), metrics=["accuracy"], optimizer="sgd")
hist_ = model.fit(X_random, y_random, batch_size=batch_size, epochs=epochs)
hist_ = hist_.history
ax[0].plot(hist_["accuracy"], label="Random Sampler")
ax[1].plot(hist_["loss"], label="Random Sampler")

plt.legend()
plt.show()
