# instruction
# For this exercise, you will use logistic regression and neural networks to
# recognize handwritten digits (from 0 to 9).
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def get_hundred(X):
    res = []
    for i in range(100):
        res.append(X[i])
    return res

def generate_image(samples):
    a = 0
    res = []
    # set the format of the image
    for i in range(10):
        temp = []
        for j in range(10):
            temp.append(np.reshape(samples[a], (20, 20)))
            a += 1
        res.append(temp)
    return res

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
for i in range(10):
    for j in range(10):
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.imshow(images[i][j], cmap="Greys_r")
        plt.xticks([])
        plt.yticks([])
plt.show()
