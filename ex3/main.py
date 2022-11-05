from scipy.io import loadmat
import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np
import random

# instruction
# For this exercise, you will use logistic regression and neural networks to
# recognize handwritten digits (from 0 to 9).

# get one hundred random rows from the X
def get_hundred(X):
    res = []
    for i in range(100):
        r = random.randint(0, 5000)
        res.append(X[r])
    return res

# transfer the sample rows into 20 * 20 matrix
def generate_image(samples):
    a = 0
    res = []
    # set the format of the image
    for i in range(10):
        temp = []
        for j in range(10):
            # transfer the one row into two dimension image
            image = np.reshape(samples[a], (20, 20))
            # swap the row and the collom
            # to make it easy to visualize
            image = [[row[i] for row in image] for i in range(len(image[0]))]
            temp.append(image)
            a += 1
        res.append(temp)
    return res

# show the 20 * 20 grey images
def show_images(images):
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.imshow(images[i][j], cmap="Greys_r")
            plt.xticks([])
            plt.yticks([])

# load and show 100 random images from X
def load_and_show_hundred_images(X):
    # get the first hundred samples
    samples = get_hundred(X)
    # transfer the data into the image matrix
    images = generate_image(samples)
    # show the images
    show_images(images)
    plt.show()

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# calculate the cost
def cost_function(theta, X, y, lam):
    m = len(y)
    ones = np.matrix(np.ones(m))
    ones_401 = np.matrix(np.ones(401))

    theta = np.matrix(theta).T
    H = sigmoid(X * theta)
    temp = np.multiply(y, np.log(H)) + np.multiply(ones - y, np.log(ones - H))

    J = ones * temp * -1 / m + (ones_401 * np.multiply(theta, theta) - theta[0, 0] * theta[0, 0]) * lam / 2 / m

    return J[0, 0]

# calculate the gradient
def gradient(theta, X, y, lam):
    j = len(theta)
    m = len(y)
    theta = np.matrix(theta).T
    H = sigmoid(X * theta)

    # calculate the gradient
    grad = ((H - y).T * X) / m

    # print(len(grad))
    # print(len(grad[0]))
    # print(type(grad))

    for i in range(1, j):
        grad[0, i] += lam * theta[i] / m

    # print(len(grad))
    # print(len(grad[0]))

    return grad[0]

# calculate the multi class classifier
def one_vs_all(X, y, num_labels, lam):
    res = []
    theta = np.matrix(np.zeros(401))

    for i in range(num_labels):
        temp_y = transfer_y(y, i + 1)
        temp_res = op.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, temp_y, lam))
        res.append(temp_res[0])

    return res

# transfer the y from multi class to binary class
def transfer_y(y, label):
    res = []

    for i in y:
        if i == label:
            res.append(1)
        else:
            res.append(0)

    return res

# add a colom of one to X
def transfer_X(X):
    newcols = np.ones(5000).T
    res = np.insert(X, 0, newcols, axis=1)

    return np.matrix(res)

# preidcate which class the image belong to
def predicate(X, theta_list):
    res_label = -1
    max_h = 0

    for i in range(len(theta_list)):
        theta = theta_list.get(i + 1)
        temp_h = sigmoid(X * theta.T)
        if temp_h > max_h:
            max_h = temp_h
            res_label = i + 1

    return (res_label, max_h)


###### show the images ########
# load the all the data from the file
file = loadmat("ex3data1.mat")
# get the 5000 images
X = file.get("X")
# get the result of the image
y = file.get("y")


###### learn the parameters ##########
X = transfer_X(X)
y = y.T[0]
theta_list = one_vs_all(X, y, 10, 1)
print(theta_list)