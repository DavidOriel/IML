import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def fit_linear_regression(X, y):
    #question 9
    """
    fined solution u for linear regression problem
    :param X: matrix that the samples are representing as columns
    :param y: the response vector
    :return: the solution for the linear regression problem, and the singular values of the svd
    """
    w = np.dot(np.linalg.pinv(X).T, y)
    sigma = np.linalg.svd(X, compute_uv=False)
    return w, sigma


def predict(X, w):
    #question 10
    """
    predict the response vector y
    :param X: matrix that the samples are representing as columns
    :param w: w vector at the Xw = y prob
    :return: y
    """
    return np.dot(X.T, w)


def mse(y, y0):
    #question 11
    """
    calculate the mse between a real vector and predicted vector
    :param y: real vector
    :param y0: predicted vector
    :return: the mse between them
    """
    return np.nanmean(np.square(np.subtract(y, y0)))


def load_data(path):
    #question 12 -13
    """
    load a data from a given path, filtering the data from non important features and create dummies for features
    that cannot be estimated by their integer values,
     then the function will extract the price column and after that the function will
    stacks arrays of ones to the data for the linear regression process.
    :param path: the path to the data file.
    :return: filtered and reorgenized numpy matrix, and the price column y.
    """
    df = pd.read_csv(path)
    df.to_numpy()
    dummies = pd.get_dummies(df.zipcode, drop_first=True)
    merged = pd.concat([df, dummies], axis="columns")
    y = merged["price"]
    y = y.to_numpy()
    y = np.nan_to_num(y)
    merged["date"] = merged["date"].str.replace("T000000", "")
    merged = merged.fillna(0)
    merged["date"] = merged["date"].astype(int)
    print(merged["date"])
    df_final = merged.drop(["zipcode", "price", "lat", "long", "id"], axis="columns")
    df_final = np.nan_to_num(df_final)
    df_final = np.transpose(df_final)
    columns = len(df_final[0])
    ones = np.array([1] * columns)
    df_final = np.vstack([ones, df_final])
    return df_final, y


def plot_singular_values(sin):
    #question 14
    """
    plotting an array of singular values
    :param sin: array of singular values
    :return: none
    """
    sin = np.sort(sin)
    sin = sin[::-1]
    plt.scatter(np.arange(len(sin)), sin)
    plt.title("the singular values of the fitted data")
    plt.xlabel("number")
    plt.ylabel("value")
    plt.show()


def putting_together_1():
    #question 15
    """
    given a path, the function loads the data and and after getting the singular values from it, plotting them
    :return: returns the data Matrix , and the price column y
    """
    X, y = load_data("C:/Users/david/Desktop/machine learning/ex2/kc_house_data.csv")
    w, sigma = fit_linear_regression(X, y)
    plot_singular_values(sigma)
    return X, y


def putting_together_2(X, y):
   #question 16
    """
    function that fitting a model and then test it over the data, after that it will plot the mse over the test set as
    function of percentage.
    :param X: the data matrix
    :param y: the price column
    :return: none
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X.T, y, test_size=0.25, random_state=0)
    w_ar = []
    predicted_y = []
    mse_ar = []
    for p in range(1, 101):
        rows_x = round((p / 100) * len(X_train))
        rows_y = round((p / 100) * len(X_train))
        x_p = X_train[0:rows_x]
        y_p = Y_train[0:rows_y]
        w, sig = fit_linear_regression(x_p.T, y_p)
        w_ar.append(w)
        y_pre = predict(X_test.T, w)
        predicted_y.append(y_pre)
        mse_ar.append(mse(Y_test, y_pre))
    plt.scatter(np.arange(100), mse_ar)
    plt.title("MSE as percentage over training samples")
    plt.xlabel("value")
    plt.ylabel("number")
    plt.show()


def pearson_corelation(v1, v2):
    """
    calculate the pearson curelation between 2 vectors.
    :param v1: vector 1.
    :param v2: vector 2.
    :return: the pearson correlation.
    """
    return np.divide(np.cov(v1, v2)[0][1] ,np.dot(np.nanstd(v1) ,np.nanstd(v2)))

def feature_evaluation(X, y):
    #question 17
    """
    function that plots for any non categorical feature, a graph that represent the response value (the prices)
    as function of the feature.
    :param X: A designed matrix.. i assumed that the matrix here is already filtered after calling the function load
    data. (it wasnt clear how the parameters are designed matrix and price column y and the column y can come only after
    the manipulation over the designed matrix)...
    :param y: the price column
    :return:none
    """
    a= ["date","bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition",
            "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]
    # print(y)
    y = np.nan_to_num(y)
    for i in range(len(a)):
        plt.subplot(4,4,i+1)
        feature = X[i+1]
        np.nan_to_num(feature)
        pc = pearson_corelation(feature.T,y)
        plt.scatter( feature,y, s=10, c="b")
        plt.title("house price as function of\n"+a[i]+".\n pc is :"+str(round(pc,3)),fontdict={"fontsize":8})
        plt.xlabel("feature value",fontdict={"fontsize":7})
        plt.ylabel("price value",fontdict={"fontsize":7})
    plt.show()


X, y = putting_together_1()
putting_together_2(X, y)
feature_evaluation(X,y)

