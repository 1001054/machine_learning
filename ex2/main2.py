import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

# title:
# Suppose you are the product manager of the factory and you have the
# test results for some microchips on two different tests. From these two tests,
# you would like to determine whether the microchips should be accepted or
# rejected.

# get the accepted and rejected training set
def get_data():
    pass_score1 = []
    pass_score2 = []
    fail_score1 = []
    fail_score2 = []

    with open(r"ex2data2.txt", "r") as f:
        for line in f.readlines():
            temp = line.split(",")
            if int(temp[2]) == 1:
                pass_score1.append(float(temp[0]))
                pass_score2.append(float(temp[1]))
            else:
                fail_score1.append(float(temp[0]))
                fail_score2.append(float(temp[1]))

    return (pass_score1, pass_score2, fail_score1, fail_score2)

# get all the training data
def get_raw_data():
    score1 = []
    score2 = []
    is_admitted = []

    with open(r"ex2data2.txt", "r") as f:
        for line in f.readlines():
            temp = line.split(",")
            score1.append(float(temp[0]))
            score2.append(float(temp[1]))
            is_admitted.append(int(temp[2]))

    return (score1, score2, is_admitted)

# show the samples
def show_samples():
    (pass_score1, pass_score2, fail_score1, fail_score2) = get_data()

    plt.scatter(pass_score1, pass_score2, marker="+", color="black")
    plt.scatter(fail_score1, fail_score2, marker="o", color="yellow")
    plt.xticks([-1, -0.5, 0, 0.5, 1, 1.5])
    plt.yticks([-0.8, -0.4, 0, 0.4, 0.8, 1.2])
    plt.xlabel("Microchip test1")
    plt.ylabel("Microchip test2")

def show_decision_boundary(theta):
    u = np.linspace(-1, 1.5, 250)
    v = np.linspace(-1, 1.5, 250)
    z = np.zeros([len(u), len(v)])
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = theta * np.matrix(map_feature(u[i], v[j])).T
    plt.contour(u, v, z, 0)
    plt.show()

# map all features of six power
def map_feature(x1, x2):
    res = []

    for i in range(0, 7):
        for j in range(0, i + 1):
            res.append(np.power(x1, j) * np.power(x2, (i - j)))

    return res

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# Compute cost for logistic regression with regularization
def cost_function(theta, X, y, lam):
    m = len(y)
    ones = np.matrix(np.ones(m))
    ones_28 = np.matrix(np.ones(28))

    print(len(X))
    print(len(X[0]))
    H = sigmoid(theta * X)
    temp = np.multiply(y, np.log(H)) + np.multiply(ones - y, np.log(ones - H))
    J = temp * ones.T * -1 / m + (np.multiply(theta, theta) * ones_28.T - theta[0] * theta[0]) * lam / 2 / m

    return J

# Compute gradient for logistic regression with regularization
def gradient(theta, X, y, lam):
    j = len(theta)
    m = len(y)
    H = sigmoid(theta * X)

    # calculate the gradient
    grad = ((H - y) * X.T) / m
    for i in range(1, j):
        grad[0, i] += lam * theta[i] / m

    return grad[0]


(x1, x2, y) = get_raw_data()
m = len(y)
X = np.matrix(map_feature(x1, x2))
# the init theta
theta = np.zeros(28)
# regularization parameter
# to penalize the theta parameters from 1 to m
lam = 1
# an optimization solver that finds the minimum of an unconstrained function
result = op.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y, lam))
final_theta = result[0]
show_samples()
show_decision_boundary(final_theta)