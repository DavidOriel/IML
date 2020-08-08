import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from plotnine import *
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics
import matplotlib.pyplot as plt


CV = 5

def f(x):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)

def create_polynom(low,high,k):
    x = np.linspace(low, high, k)
    y = f(x)
    return x, y

def plot_polynom(x,y):

     g=(ggplot(DataFrame({'x':x, 'y':y}),aes(x='x', y='y'))+geom_point(color = "green", size =.9)+\
     geom_line(size =.5,alpha =.5))
     print(g)

def simple_linear_reg(x, y):
    X = x.reshape(-1,1)
    print(x)
    model = LinearRegression().fit(X,y)
    p=(ggplot(DataFrame({'x': x, 'y': y, 'y_hat': model.predict(X)}),
            aes(x='x', y='y')) +
     geom_line(size=.5, alpha=.5) +
     geom_point(color="blue", size=.7) +
     geom_point(aes(x='x', y='y_hat'), color="orange", size=.7) +
     labs(title="Linear Regression Fitting\n" + r"$R^2={:.2E}$".format(model.score(X, y))))
    print(p)


def fit_poly_linear_reg(x,y):
    dfs = []
    X = x.reshape(-1,1)
    for k in range(1, 10):
        model = make_pipeline(PolynomialFeatures(k), LinearRegression())
        model.fit(X, y)
        y_hat = model.predict(X)
        dfs.append(DataFrame({'x': x, 'y': y, 'y_hat': y_hat, 'k': k,
                              'mse': r"$MSE={:.2E}$".format(np.mean((y - y_hat) ** 2))}))

    dfs = pd.concat(dfs)

    p= (ggplot(dfs, aes(x='x', y='y')) +
        geom_line(size=.5, alpha=.5) +
        geom_point(color="blue", size=.3, alpha=.5) +
        geom_line(aes(x='x', y='y_hat'), color="orange", size=.7) +
        geom_text(aes(label='mse'), x=0, y=-15000, size=6) +
        facet_wrap("~k"))
    print(p)

def add_noise(data , factor):
    """
        adding normal noise with zero mean and factor times
        the standard deviation (of the given data) (factor*std)
        :param data: The samples to add noise to
        :param factor: The factor to multiply the standard deviation by
        :return: An array of same size with the added normal noise
        """
    k = np.std(data) * factor
    noise = np.random.normal(0, k, len(data))
    return noise


def generate_samples(m,sigma):
    X = np.random.uniform(-3.2,2.2, m)
    y= f(X)
    epsilon = np.random.normal(0,sigma**2,m)
    y= y + epsilon
    D = X[:1000], y[:1000]
    T= X[1000:], y[1000:]
    return D,T

def calculate_ERM(D, d):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(D[0].reshape(-1,1), D[1])
    y_hat = model.predict(D[0].reshape(-1,1))
    ERM= np.mean((D[1] - y_hat)**2)
    return model, ERM

def find_h_star(D):
    S = D[0][:500], D[0][:500]
    V = D[1][500:], D[1][500:]
    model_array= []
    for i in range(1,16):
        model = make_pipeline(PolynomialFeatures(i+1), LinearRegression())
        model.fit(S[0].reshape(-1,1),S[1])
        model_array.append(model)
    y_hat = model_array[0].predict(V[0].reshape(-1,1))
    h_star = model_array[0]
    y = V[1]
    loss = np.mean((y-y_hat)**2)

    for i in range(1,15):
        y_hat = model_array[i].predict(V[0].reshape(-1,1))
        if np.mean((y-y_hat)**2)<loss:
            loss = np.mean((y-y_hat)**2)
            h_star = model_array[i]
    return h_star

def k_fold(train, test, k):
    kf = KFold(n_splits=k)
    val_error, train_error = np.zeros(15),np.zeros(15)
    for i in range(1,16):
        for train_index, test_index in kf.split(train[0]):
            model = make_pipeline(PolynomialFeatures(i), LinearRegression())
            S = train[0][train_index], train[1][train_index]
            V= train[0][test_index], train[1][test_index]
            model.fit(S[0].reshape(-1,1), S[1])
            y_hat = model.predict(V[0].reshape(-1,1))
            val_error[i-1] += np.mean((V[1] - y_hat)**2)
            y_hat = model.predict(S[0].reshape(-1,1))
            train_error[i-1] += np.mean((S[1] - y_hat)**2)
    arr = np.arange(1,15)+1
    g = (ggplot(DataFrame({'x':np.arange(15)+1, 'train_error':val_error,'val_error':train_error}),
           aes(x="x", y = "train_error"))+geom_line(aes(x='x', y='val_error')))

    d = np.argmin(val_error)
    model, ERM = calculate_ERM(train, d)
    y_hat = model.predict(test[0].reshape(-1,1))
    ERM_test = np.mean((test[1]-y_hat[0])**2)
    g+= labs(title = "the ERM is " + str(ERM_test))
    print(g)

def run_all():
    f = lambda x: (x - 2) * (x + 3) * (x - 4) * (x - 5) * (x + 7) - 20
    f_str = r"$\left(x-2\right)\left(x+3\right)\left(x-4\right)\left(x-5\right)\left(x+7\right)-20$"

    x = np.linspace(-9, 10, 30)
    y = f(x)

    (ggplot(DataFrame({'x': x, 'y': y}), aes(x='x', y='y')) +
     geom_line(size=.5, alpha=0.5) +
     geom_point(color="blue", size=.7) +
     labs(title="Polynomial Data\n" + f_str))
    x, y =create_polynom(-10 , 10, 30)
    g = plot_polynom(x,y)
    print(g)
    simple_linear_reg(x,y)
    fit_poly_linear_reg(x,y)
    noise = add_noise(x, 5)
    x_noisy = x+noise
    simple_linear_reg(x_noisy,y)

    D, T = generate_samples(1500, 1)
    h_star_1 = find_h_star(D)
    k_fold(D,T, 5)

    D, T = generate_samples(1500 ,5)
    h_star_5 = find_h_star(D)
    k_fold(D,T , 5)
    return


################################################################################
num_evaluations = 50
alphas_list = np.linspace(0.001, 2, num=num_evaluations)
# alphas_list = [1e-10,1e-5, 1e-2, 1, 5,10]
alphas = {"alpha": alphas_list}
best_model_errors_list =[]
best_alphas = []

K= 5
X,y = datasets.load_diabetes(return_X_y=True)
num_evaluations = 50
alpha_range = np.linspace(0.001, 2, num=num_evaluations)

def generate_samples2():
    xtrain = X[:50]
    ytrain = y[:50]
    xtest = X[50:]
    ytest = y[50:]
    return xtrain, ytrain,xtest,ytest

def train_ridge(X, y, alpha):
    clf = Ridge(alpha=alpha)
    clf.fit(X, y)
    return clf

def train_lasso(X, y, alpha):
    clf = Lasso(alpha=alpha)
    clf.fit(X, y)
    return clf

def train_lr(X, y):
    clf = LinearRegression()
    clf.fit(X, y)
    return clf

def validate(X, y, clf):
    return np.mean((clf.predict(X) - y)**2)

# 'neg_mean_squared_error'
def Kfold_models(xtrain,ytrain,xtest,ytest, K,alphas,model):
    regressor = GridSearchCV(model, alphas, cv=K,return_train_score=True,scoring = 'neg_mean_squared_error').fit(xtrain, ytrain)
    validation_error = np.zeros(len(alphas))
    best_alpha = regressor.best_params_
    best_model = regressor.best_estimator_
    # print(regressor.grid_scores_[0])
    val_error = 1- regressor.cv_results_.get("mean_test_score")
    train_error = 1- regressor.cv_results_.get("mean_train_score")
    g = (ggplot(DataFrame({'x': alphas_list, 'train_error': val_error, 'val_error': train_error}),
                aes(x="x", y="train_error")) +geom_line(color="blue")+ geom_line(aes(x='x', y='val_error')))
    y_hat = best_model.predict(xtest)
    best_model_error = metrics.mean_squared_error(ytest,y_hat)
    best_model_errors_list.append(best_model_error)
    best_alphas.append(best_alpha)
    print(g)


# def their_code():
#
#     train_size = 60
#     kfold = 4
#     train_x = X[:train_size, :]
#     train_y = y[:train_size]
#     test_x = X[train_size:, :]
#     test_y = y[train_size:]
#     num_evaluations = 50
#     avg_train_err_lasso = np.zeros(num_evaluations)
#     avg_validation_err_lasso = np.zeros(num_evaluations)
#     groups = np.remainder(np.arange(train_y.size), kfold)
#     for k in range(kfold):
#         S_k_x = train_x[groups != k]
#         S_k_y = train_y[groups != k]
#         V_k_x = train_x[groups == k]
#         V_k_y = train_y[groups == k]
#         h_d = [train_lasso(S_k_x, S_k_y, alpha) for alpha in alphas_list]
#         loss_train_d = [validate(S_k_x, S_k_y, clf) for clf in h_d]
#         loss_validation_d = [validate(V_k_x, V_k_y, clf) for clf in h_d]
#         avg_train_err_lasso += np.array(loss_train_d) / kfold
#         avg_validation_err_lasso += np.array(loss_validation_d) / kfold
#     plt.plot(alphas_list, avg_train_err_lasso, label='train: lasso')
#     plt.plot(alphas_list, avg_validation_err_lasso, label='validation: lasso')
#     plt.legend()
#     plt.show()

def run_all2():
    lasso = Lasso()
    xtrain, ytrain, xtest, ytest = generate_samples2()
    Kfold_models(xtrain,ytrain,xtest,ytest,5,alphas, lasso)
    # their_code()

if __name__ == '__main__':
    run_all2()








