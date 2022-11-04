from scipy.io import loadmat
import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np
import random

# Instruction
# Your goal is to implement the feedforward
# propagation algorithm to use our weights (ex3weights) for prediction.

# load the data from the file
weights = loadmat("ex3weights.mat")
