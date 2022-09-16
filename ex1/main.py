from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# introduction
# In this part of this exercise, you will implement linear regression with one
# variable to predict profits for a food truck. Suppose you are the CEO of a
# restaurant franchise and are considering different cities for opening a new
# outlet. The chain already has trucks in various cities and you have data for
# profits and populations from the cities.

# get the data from the txt
def get_data():
    x = []
    y = []

    with open("ex1data1.txt", "r") as f:
        for line in f.readlines():
            nums = line.split(",")
            # print(line)
            x.append(float(nums[0]))
            y.append(float(nums[1].replace("\n", "")))

    return (x, y)

# the cost function
# the x is the matrix of ones and the house size (x)
# the y is the house price (y)
# the theta is the parameter matrix of the linear model
def compute_cost(x, y, theta):
    m = len(y)
    temp = theta * x - np.mat(y)
    # elements square
    temp = np.multiply(temp, temp)
    # sum
    temp = temp * np.mat(np.ones(m)).T
    res = temp[0, 0] / 2 / m
    return res

# use the gradient_descent to calculate the theta
# the alpha is the learning rate
def gradient_descent(x, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        print(compute_cost(x, y, theta))
        # the Partial derivative
        temp0 = theta[0, 0] - alpha / m * ((theta * x - np.mat(y)) * x[0].T)[0, 0]
        temp1 = theta[0, 1] - alpha / m * ((theta * x - np.mat(y)) * x[1].T)[0, 0]
        theta = np.mat([temp0, temp1])

    return theta

# show the samples and the linear regression
def show_plot():
    (x, y) = get_data()
    # show the samples
    plt.scatter(x, y, marker="x", color="r", label="Training data")
    plt.xticks(range(int(min(x)) - 1, int(max(x)) + 1))
    plt.yticks(range(int(min(y)) - 1, int(max(y)) + 1))
    plt.xlabel("Profit in $10,000s")
    plt.ylabel("Population of City in 10,000s")
    show_linear_regression(x, y)
    plt.legend(loc="lower right")
    plt.show()

# show the linear regression
def show_linear_regression(x, y):
    # initialize
    m = len(x)
    theta = np.mat([0, 0])
    x = np.mat([np.ones(m), x])
    # calculate the theta
    theta = gradient_descent(x, y, theta, 0.01, 1500)
    x2 = np.arange(x[1].min(), x[1].max())
    y2 = theta[0, 1] * x2 + theta[0, 0]
    plt.plot(x2, y2, label="Linear regression")

# =============visualizing J(theta0, theta1)=================

def j(theta0, theta1):
    (x, y) = get_data()
    m = len(x)
    x = np.mat([np.ones(m), x])
    theta = np.mat([theta0, theta1])
    return compute_cost(x, y, theta)

# the points of good view
def get_figure_data():
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)
    J = np.zeros((100, 100))
    for m in range(len(theta0)):
        for n in range(len(theta1)):
            J[m, n] = j(theta0[m], theta1[n])

    return (theta0, theta1, J)

def show_contour():
    (theta0, theta1, J) = get_figure_data()

    plt.contourf(theta0, theta1, J, 15, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(theta0, theta1, J, 15, colors='black')
    plt.clabel(C, inline=1, fontsize=10)
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.show()

def show_surface():
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    (theta0, theta1, J) = get_figure_data()

    ax.plot_surface(theta0, theta1, J)
    ax.set_zlim(0, 800)
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.show()

