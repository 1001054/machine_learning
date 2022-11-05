from scipy.io import loadmat
import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np
import random

# Instruction
# Your goal is to implement the feedforward
# propagation algorithm to use our weights (ex3weights) for prediction.

# get five random rows from the X
def get_five(X, y):
    a = []
    b = []

    for i in range(5):
        r = random.randint(0, 5000)
        a.append(X[r])
        b.append(y[r])

    return (a, b)

# add one column of ones for X
def add_one_cols(X):
    n = len(X)
    newcols = np.ones(n).T
    res = np.insert(X, 0, newcols, axis=1)
    return res

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# return the max index from the hypothesis arrays
def get_max(hs):
    res = []

    if type(hs) == type(np.matrix([])):
        hs = hs.tolist()
    for h in hs:
        max = -100
        index = 1

        for i, num in enumerate(h):
            if max < num:
                max = num
                index = i + 1
        res.append(index)

    return res

# predicate the result of the samples
# by using neural network
def predicate(X, theta1, theta2):
    temp = sigmoid(add_one_cols(X) * theta1.T)
    res = sigmoid(add_one_cols(temp) * theta2.T) * 100

    return res

# load the data from the file
weights = loadmat("ex3weights.mat")
theta1 = np.matrix(weights.get("Theta1"))
theta2 = np.matrix(weights.get("Theta2"))
data = loadmat("ex3data1.mat")
X = data.get("X")
y = data.get("y")

(X, y) = get_five(X, y)
X = np.matrix(X)
y = np.matrix(y).T
hs = predicate(X, theta1, theta2)
res = get_max(hs)

print("The hypothesis:")
print(hs)
print("The predicated results:")
print(res)
print("")
print("The right results:")
print(y.tolist()[0])


