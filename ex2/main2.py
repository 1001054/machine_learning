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


show_samples()