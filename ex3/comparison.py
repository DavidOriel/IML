import numpy as np
from plotnine import *
from pandas import DataFrame
import matplotlib.pyplot as plt
from models import *

AR = [5, 10, 15, 25, 70]
w = np.array([0.1, 0.3, -0.5])
K = 1000
perceptron = Perceptron()
svm = SVM()
lda = LDA()
model_list = [perceptron, svm, lda]


def draw_points(m):
    """
    draw n points
    :param m: how many points to draw
    :return: x as data, with the first column as ones and y as the classification of the data
    """
    I2 = np.identity(2)
    x = np.random.multivariate_normal([0, 0], I2, m)
    X0 = np.ones((m, 1))
    x = np.hstack((X0,x))
    y = np.sign(np.dot(x, w))
    return x, y



for m in AR:
    x, y = draw_points(m)
    while 1 not in y or -1 not in y:
        x, y = draw_points(m)
    xx = np.linspace(np.min(x), np.max(x))
    yy = np.linspace(0.6 * np.min(x) + 0.2, 0.6 * np.max(x) + 0.2)  #
    perceptron.fit(x, y)
    wp = perceptron.model
    a = -w[1] / w[2]
    yp = a * xx - w[0] / w[2]
    svm.fit(x,y)
    w = svm.model.coef_[0]
    a = -w[0] / w[1]
    ys = a * xx - (svm.model.intercept_[0]) / w[1]
    df = DataFrame({'x1': x[:, 1], 'x2': x[:, 2], "y": y})
    plot = (ggplot() + geom_point(df, aes(x="x1", y="x2", color="factor(y)")))
    plot += (geom_line(aes(x="x1", y="y1",color = "true line"),
                       data=DataFrame({'x1': xx, 'y1': yy,"true line":"tl"})))
    plot += (geom_line(aes(x="x1", y="y1",color = "perceptron_line"),
                       data=DataFrame({'x1': xx, 'y1': yp,"perceptron_line":"pl"})))
    plot += (geom_line(aes(x="x1", y="y1",color = "svm_line"),
                       data=DataFrame({'x1': xx, 'y1': ys, "svm_line":"sv"})))
    print(plot)
    ggsave(plot,filename = "plot"+str(m))
perceptron = Perceptron()
svm = SVM()
lda = LDA()


def test():
    """
    test the perceptron, lda,svm classification over a data
    :return: none
    """
    meanarr1 = np.array([])
    meanarr2 = np.array([])
    meanarr3 = np.array([])
    for m in AR:
        acc1 =np.array([])
        acc2 = np.array([])
        acc3 = np.array([])
        for i in range(500):
            x, y = draw_points(m)
            xz, yz = draw_points(K)
            while 1 not in y or -1 not in y:
                x, y = draw_points(m)
            while 1 not in yz or -1 not in yz:
                xz, yz= draw_points(K)
            perceptron.fit(x, y)
            svm.fit(x,y)
            lda.fit(x,y)
            acc1= np.append(acc1, [perceptron.score(xz,yz).get("accuracy")])
            acc2 =np.append(acc2, [svm.score(xz,yz).get("accuracy")])
            acc3 = np.append(acc3, [lda.score(xz,yz).get("accuracy")])
        meanarr1 = np.append(meanarr1, np.mean(acc1))
        meanarr2 = np.append(meanarr2, np.mean(acc2))
        meanarr3 = np.append(meanarr3, np.mean(acc3))
    plot = ggplot()+geom_line(aes(x="x",y="y",color="perceptron"),data= DataFrame({"x":AR,"y":meanarr1,"perceptron":"perceptron" }))
    plot+= geom_line(aes(x="x",y="y",color="svm"),data= DataFrame({"x":AR,"y":meanarr2 ,"svm":"svm"}))
    plot+= geom_line(aes(x="x",y="y",color="lda"),data= DataFrame({"x":AR,"y":meanarr3,"lda":"lda" }))
    print(plot)
    ggsave(plot,filename = "plot")


test()