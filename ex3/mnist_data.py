import tensorflow as tf
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import random
from models import *
from pandas import DataFrame

logistic = Logistic()
tree = DecisionTree()
soft_svm = Soft_SVM()
k_nearest = K_nearest()

AR = [5, 10, 15, 25, 70]
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]
plt.imshow(x_train[8])
plt.show()
labaled_0 = [x_train[0], x_train[5], x_train[8]]
labeled_1 = [x_train[1], x_train[2], x_train[3]]


def rearrange_data(X):
    return np.reshape(X, (np.shape(X)[0], 784))


def get_random(x_list, y_list, m):
    inc = np.arange(y_list.shape[0])
    np.random.shuffle(inc)
    if m is not None:
        inc = inc[:m]
    random_x = x_list[inc].astype(np.int8)
    random_y = y_list[inc].astype(np.int8)
    for i in range(len(inc)):
        if random_y[i] == 0:
            random_y[i] = -1
    return random_x, random_y


meanarr1 = np.array([])
meanarr2 = np.array([])
meanarr3 = np.array([])
meanarr4 = np.array([])
for m in AR:
    x_rand, y_rand = get_random(x_train, y_train, m)
    while 1 not in y_rand or -1 not in y_rand:
        x_rand, y_rand = get_random(x_train, y_train, m)
    acc1 = np.array([])
    acc2 = np.array([])
    acc3 = np.array([])
    acc4 = np.array([])
    for i in range(50):
        x_rand = rearrange_data(x_rand)
        logistic.fit(x_rand, y_rand)
        tree.fit(x_rand, y_rand)
        # k_nearest.fit(x_rand, y_rand)
        soft_svm.fit(x_rand, y_rand)

        x_test, y_test = get_random(x_test, y_test, None)

        x_test = rearrange_data(x_test)
        acc1 = np.append(acc1, logistic.score(x_test, y_test).get("accuracy"))

        # NOTE: FAILS HERE
        # acc_k_nearest = k_nearest.score(x_test, y_test).get("accuracy")


        # acc2 = np.append(acc2, acc_k_nearest)
        acc3 = np.append(acc3, soft_svm.score(x_test, y_test).get("accuracy"))
        acc4 = np.append(acc4, tree.score(x_test, y_test).get("accuracy"))
    meanarr1 = np.append(meanarr1, np.mean(acc1))
    meanarr2 = np.append(meanarr2, np.mean(acc2))
    meanarr3 = np.append(meanarr3, np.mean(acc3))
    meanarr4 = np.append(meanarr4, np.mean(acc4))
plot = ggplot() + geom_line(aes(x="x", y="y", color="logistic"),
                            data=DataFrame({"x": AR, "y": meanarr1, "logistic": "logistic"}))
plot += geom_line(aes(x="x", y="y", color="k_nearest"),
                  data=DataFrame({"x": AR, "y": meanarr2, "k_nearest": "k_nearest"}))
plot += geom_line(aes(x="x", y="y", color="svm"), data=DataFrame({"x": AR, "y": meanarr3, "svm": "svm"}))
plot += geom_line(aes(x="x", y="y", color="tree"), data=DataFrame({"x": AR, "y": meanarr4, "tree": "tree"}))
ggsave(plot, filename="plot1111")
