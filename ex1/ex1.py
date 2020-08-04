
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T

def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


# question 11-15
plot_3d(x_y_z)
scaling_matrix = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
mult = np.dot(scaling_matrix, x_y_z)
plot_3d(mult)
orth = get_orthogonal_matrix(3)
mult1 = np.dot(orth, mult)
plot_3d(mult1)
plot_2d(x_y_z[[0, 1], :])
x_y_z[x_y_z >= 0.1] = 0.09
x_y_z[x_y_z <= -0.4] = -0.39
plot_2d(x_y_z[[0, 1], :])

# question 16

Experiments = 100000
Trials = 1000
data = np.random.binomial(1, 0.25, (Experiments, Trials))
epsilon = np.array([0.5, 0.25, 0.1, 0.01, 0.001])

for i in range(5):
    plt.plot(np.cumsum(data[i]) / np.arange(1, Trials + 1), label="the #" + str(i) + " experiment")

plt.legend()
plt.show()

i = 0
for eps in epsilon:
    i += 1
    plt.subplot(3, 2, i)
    chebyshev = 1 / (4 * ((eps ** 2) * np.arange(1, Trials + 1)))
    ones = np.array([1] * 1000)
    min1 = np.min(np.vstack((chebyshev, ones)), axis=0)
    plt.plot(min1, label="chebyshev")
    hoeffding = 2 * np.exp(-2 * np.arange(1, Trials + 1) * eps ** 2)
    min2 = np.min(np.vstack((hoeffding, ones)), axis=0)
    plt.plot(min2, label="hoeffding")
    plt.title("epsilon" + str(eps), fontsize=4)
    plt.plot(
        np.sum(
            np.abs(
                (
                        np.cumsum(data, axis=1)
                        / (np.arange(1, Trials + 1))
                ) - 0.25
            ) >= eps, axis=0
        ) / Experiments,
        label="part C")
plt.legend(bbox_to_anchor=(1.5, 1), loc=2, borderaxespad=3)
plt.show()
