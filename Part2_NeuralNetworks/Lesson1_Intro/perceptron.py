#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 00:00:27 2021

@author: pkjha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for (point, label) in zip(X, y):
        y_hat = prediction(point, W, b)
        mislabeled = (y_hat != label)
        if mislabeled:
            if y_hat == 1:
                W[0] -= learn_rate*point[0]
                W[1] -= learn_rate*point[1]
                b -= learn_rate
            if y_hat == 0:
                W[0] += learn_rate*point[0]
                W[1] += learn_rate*point[1]
                b += learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

###################################################################
"""
Read and plot raw data
"""
data = pd.read_csv('data.csv').to_numpy()
data_pos = data[data[:,2] == 1]
data_neg = data[data[:,2] == 0]
plt.figure(1)
plt.scatter(data_pos[:, 0], data_pos[:, 1], color = 'b')
plt.scatter(data_neg[:, 0], data_neg[:, 1], color = 'r')


# Run the perceptron algorithm
#x_end_points = np.array([min(data[:,0]), max(data[:,0])])
x_end_points = np.array([0, 1])
boundary_lines = trainPerceptronAlgorithm(X = data[:, 0:2], y=data[:,2].astype(int), learn_rate = 0.01, num_epochs = 25)

# Plot the classification lines
for line_count, boundary_line in enumerate(boundary_lines):
    y_line = boundary_line[0]*x_end_points + boundary_line[1] # y = mx + c
    if line_count == len(boundary_lines) - 1:
        plt.plot(x_end_points, y_line, color = 'k')
    else:
        plt.plot(x_end_points, y_line, '-.g')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])