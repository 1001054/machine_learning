import matplotlib.pyplot as plt
import numpy as np

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
    plt.show()

# the logistic regression hypothesis is a sigmoid function whose shape is 'S'
# For large positive values of x, the sigmoid should be close to 1, while for large
# negative values, the sigmoid should be close to 0. Evaluating sigmoid(0) should
# give you exactly 0.5.
def sigmoid(z):
    res = []

    for num in z:
        res.append(1 / (1 + np.exp(-num)))

    return res

# the cost function of the logistic regression
def cost_function(theta, X, y):
    (score1, score2, )
    return

