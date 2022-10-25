from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import random

# instruction
# For this exercise, you will use logistic regression and neural networks to
# recognize handwritten digits (from 0 to 9).

# get one hundred random row from the X
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

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# calculate the cost
def cost_function(theta, X, y, lam):
    m = len(y)
    ones = np.matrix(np.ones(m))
    ones_5000 = np.matrix(np.ones(5000))

    H = sigmoid(theta * X)
    temp = np.multiply(y, np.log(H)) + np.multiply(ones - y, np.log(ones - H))
    J = temp * ones.T * -1 / m + (np.multiply(theta, theta) * ones_5000.T - theta[0] * theta[0]) * lam / 2 / m

    return J

# calculate the gradient
def gradient(theta, X, y, lam):
    j = len(theta)
    m = len(y)
    H = sigmoid(theta * X)

    # calculate the gradient
    grad = ((H - y) * X.T) / m
    for i in range(1, j):
        grad[0, i] += lam * theta[i] / m

    return grad[0]


# load the all the data from the file
file = loadmat("ex3data1.mat")
# get the 5000 images
X = file.get("X")
# get the result of the image
y = file.get("y")
# get the first hundred samples
samples = get_hundred(X)
# transfer the data into the image matrix
images = generate_image(samples)
# show the images
show_images(images)
plt.show()
