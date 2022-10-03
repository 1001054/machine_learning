import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

# Multivariate linear regression

# The title:
# Suppose that you are the administrator of a university department and
# you want to determine each applicant’s chance of admission based on their
# results on two exams. You have historical data from previous applicants
# that you can use as a training set for logistic regression. For each training
# example, you have the applicant’s scores on two exams and the admissions
# decision.

# get the pass and failed training set
def get_data():
    pass_score1 = []
    pass_score2 = []
    fail_score1 = []
    fail_score2 = []

    with open(r"ex2data1.txt", "r") as f:
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

    with open(r"ex2data1.txt", "r") as f:
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
    plt.xticks(range(30, 101, 10))
    plt.yticks(range(30, 101, 10))
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

# show the decision boundary
def show_decision_boundary(theta):
    x = np.linspace(30, 90)
    y = (theta[0] + theta[1] * x) / theta[2] * -1
    plt.plot(x, y)
    plt.show()

# the logistic regression hypothesis is a sigmoid function whose shape is 'S'
# For large positive values of x, the sigmoid should be close to 1, while for large
# negative values, the sigmoid should be close to 0. Evaluating sigmoid(0) should
# give you exactly 0.5.
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

# the cost function of the logistic regression
# this function will return the cost and the gradient
def cost_function(theta, X, y):
    m = len(y)
    ones = np.matrix(np.ones(m))

    # calculate the cost
    H = sigmoid(theta * X)
    temp = np.multiply(y, np.log(H)) + np.multiply(ones - y, np.log(ones - H))
    J = temp * ones.T / m * -1

    return J

# to gradient decient the theta just one time
def gradient(theta, X, y):
    m = len(y)
    H = sigmoid(theta * X)

    # calculate the gradient
    grad1 = ((H - y) * X[0].T)[0, 0] / m
    grad2 = ((H - y) * X[1].T)[0, 0] / m
    grad3 = ((H - y) * X[2].T)[0, 0] / m

    return (grad1, grad2, grad3)

# calculate the probability of pass the exam
def predicate(theta, score):
    return sigmoid(theta[0] + theta[1] * score[0] + theta[2] * score[1])


(score1, score2, isadmitted) = get_raw_data()
m = len(score1)
X = np.matrix([np.ones(m), score1, score2])
theta = [0, 0, 0]

# an optimization solver that finds the minimum of an unconstrained function
result = op.minimize(fun=cost_function, x0=theta, args=(X, isadmitted), method='TNC', jac=gradient)
print(result)

# predicate a sample
theta = result.get("x")
# print(predicate(theta, (45, 85)))

# show the diagram
show_samples()
show_decision_boundary(theta)




