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
    plt.show()

# map all features of six power
def map_feature(x1, x2):
    res = []

    for i in range(0, 7):
        temp = []
        for j in range(0, i + 1):
            temp.append(np.power(x1, j) * np.power(x2, (i - j)))
        res.append(temp.copy())

    return res

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# Compute cost for logistic regression with regularization
def cost_function(theta, X, y, lam):
    m = len(y)
    ones = np.matrix(np.ones(m))

    H = sigmoid(theta * X)
    temp = np.multiply(y, np.log(H)) + np.multiply(ones - y, np.log(ones - H))
    J = temp * ones.T * -1 / m + (np.multiply(theta, theta) * ones.T - theta[0] * theta[0]) * lam / 2 / m

    return J

# Compute gradient for logistic regression with regularization
def gradient(theta, X, y, lam):
    j = len(theta)
    m = len(y)
    H = sigmoid(theta * X)

    # calculate the gradient
    grad = ((H - y) * X.T) / m
    for i in range(1, j + 1):
        grad[i] += lam * theta[i] / m

    return grad

(x1, x2, y) = get_raw_data()
m = len(y)
X = np.matrix(map_feature(x1, x2))
print(X)
# theta = np.zeros(m)
# print(cost_function(theta, X, y, 1000))