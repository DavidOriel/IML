import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

def fit_linear_regression(X, y):
    """
    fined solution u for linear regression problem
    :param X: matrix that the samples are representing as columns
    :param y: the response vector
    :return: the solution for the linear regression problem, and the singular values of the svd
    """
    w = np.linalg.pinv(X).T@ y
    sigma = np.linalg.svd(X, compute_uv=False)
    return w, sigma


def load_data(path):
    """
    function that loads the data from the path. after that the function will calculate the fitted vector w
    and will plot 2 graphs, one for the detected growth as function of days and the other for the log over
    the detected growth as function of days.
    :param path: path to the data
    :return: none
    """
    df = pd.read_csv(path)
    detected = df["detected"]
    log_det = np.array([log(det) for det in detected])
    df.insert(3,"log_detected",log_det,True)
    day_num = df["day_num"]
    day_num= day_num.to_numpy()
    day_num = np.transpose(day_num)
    columns = len(day_num)
    ones = np.array([1] * columns)
    day_num0 = np.vstack([ones, day_num])
    w = fit_linear_regression(day_num0,log_det.T)[0]
    b, m =w
    plt.scatter(day_num,log_det)
    plt.plot(day_num, m*day_num+b, label = " the fitted curve")
    plt.title("log_detected as function of day_num")
    plt.xlabel("day_num")
    plt.ylabel("log_detected")
    plt.show()
    plt.scatter(day_num,detected)
    plt.plot(day_num, np.exp(m*day_num+b), label = " the fitted curve")
    plt.title("detected as function of day_num")
    plt.xlabel("day_num")
    plt.ylabel("detected")
    plt.show()
    df.to_numpy()

load_data("C:/Users/david/Desktop/machine learning/ex2/covid19_israel.csv")