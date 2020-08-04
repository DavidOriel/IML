"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4_tools import *
from plotnine import *
from pandas import DataFrame
import matplotlib.pyplot as plt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = None
        self.samples_weights = []  # all D generated in a loop over T
        self.errors = [None] * T  # the error generated every loop

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = len(X)
        samples_weights = np.zeros(shape=(self.T + 1, m))
        samples_weights[0] = np.ones(shape=m) / m
        for t in range(self.T):
            D = samples_weights[t]
            h = self.WL(D, X, y)
            pred = h.predict(X)
            epsilon = D[(pred != y)].sum()
            wt = np.log((1 - epsilon) / epsilon) / 2
            new_D = D * np.exp(-wt * y * pred)
            new_D /= new_D.sum()
            samples_weights[t + 1] = new_D
            self.w[t] = wt
            self.h[t] = h
            self.errors[t] = epsilon
            self.D = new_D
        return self.D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        prediction = np.zeros(shape=(len(X)))
        for i in range(max_t):
            if (self.h[i] != 0):
                prediction += self.w[i] * self.h[i].predict(X)
        return np.sign(prediction)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        preds = self.predict(X, max_t)
        incorrect = (preds != y)
        return 1 - np.mean(preds == y)


# Q10
def question10_13(noise):
    X, y = generate_data(5000, noise)
    D = np.ones(shape=5000) / 5000
    T = 500
    ab = AdaBoost(DecisionStump, T)
    ab.train(X, y)
    testx, testy = generate_data(200, 0)
    error1 = []
    for i in range(T):
        error1.append(ab.error(X, y, i))
    error2 = []
    for i in range(T):
        error2.append(ab.error(testx, testy, i))

    plot = ggplot()
    plot += (geom_line(aes(x="T", y="error", color=" "), size=1,
                       data=DataFrame({'error': error1, 'T': np.arange(T), " ": "data error"})))
    plot += (geom_line(aes(x="T", y="error", color=" "), size=1,
                       data=DataFrame({'error': error2, 'T': np.arange(T), " ": "test error"})))
    plot += ggtitle("error as function of T with noise "+ str(noise))
    # ggsave(plot, filename="plot"+str(noise))
    print(plot)


    # Q11
    T = [5, 10, 50, 100, 200, 500]
    i = 1
    for t in T:
        plt.subplot(2, 3, i)
        decision_boundaries(ab, testx, testy, num_classifiers=t, weights=0.5)
        i += 1
        plt.title("plot for "+ str(t) +"classifiers")
    plt.show()

    # Q12
    tmin = np.argmin(error2)
    decision_boundaries(ab, X, y, num_classifiers=tmin)
    plt.title("minimum error is" + str(error2[tmin])+"the minimum T is "+str(tmin))
    plt.show()
    # Q13

    decision_boundaries(ab, X, y, num_classifiers=500, weights=ab.D)
    plt.title("plot for unnormalized D")
    plt.show()
    normalizeD = ab.D / np.max(ab.D) * 10
    decision_boundaries(ab, X, y, num_classifiers=500, weights=normalizeD)
    plt.title("plot for normalized D")
    plt.show()


question10_13(0)

noises = [.001, .4]
for noise in noises:
    question10_13(noise)
