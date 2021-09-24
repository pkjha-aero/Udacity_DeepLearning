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
np.random.seed(44)

epochs = 100
learnrate = 0.01

#Some helper functions for plotting and drawing lines

def plot_points(X, y):
    # Is the labelling correct?
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--', linewith = .2, alpha = 0.4):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)
"""
def stepFunction(t):
    if t >= 0:
        return 1
    return 0
"""
# Activation (sigmoid) function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

"""
def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])
"""
# Output (prediction) formula
def output_formula(features, weights, bias):
    #linear_comb = (np.matmul(features,weights) + bias)
    linear_comb = np.dot(features,weights) + bias
    """
    linear_comb = 0.0
    for i in range(len(weights)):
        linear_comb += features[i]*weights[i]
    linear_comb += bias
    """
    return sigmoid (linear_comb)

"""
def perceptronStep(X, y, W, b, learn_rate = 0.01):
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
""" 

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    y_hat = output_formula(x, weights, bias)
    d_error = y - y_hat
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias   

# Error (log-loss) formula
def error_formula(y, output):
    cross_entropy_arr = -y*np.log(output) - (1 - y)*np.log(1 - output)
    return cross_entropy_arr

"""
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
"""

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets): # For each record or data point
            #x = x.reshape(1, n_features)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        cross_entropy_arr = error_formula(targets, out)
        loss = np.mean(cross_entropy_arr)
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
            # Converting the output (float) to boolean as it is a binary classification
            # e.g. 0.95 --> True (= 1), 0.31 --> False (= 0)
            predictions = out > 0.5
            
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            plt.figure(1)
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black', 10.0, 1.0)

    # Plotting the data
    #plot_points(features, targets)
    #plt.show()

    # Plotting the error
    plt.figure(2)
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()
###################################################################
"""
Read and plot raw data
"""
"""
data = pd.read_csv('data.csv').to_numpy()
data_pos = data[data[:,2] == 1]
data_neg = data[data[:,2] == 0]
plt.figure(1)
plt.scatter(data_pos[:, 0], data_pos[:, 1], color = 'b')
plt.scatter(data_neg[:, 0], data_neg[:, 1], color = 'r')
"""
data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plt.figure(1)
plot_points(X,y)
plt.show()

# Run the perceptron algorithm
"""
x_end_points = np.array([0, 1])
boundary_lines = trainPerceptronAlgorithm(X = data[:, 0:2], y=data[:,2].astype(int), learn_rate = 0.01, num_epochs = 25)
"""
# Run the Gradient Descent Algorithm
train(X, y, epochs, learnrate, True)

"""
# Plot the classification lines
for line_count, boundary_line in enumerate(boundary_lines):
    y_line = boundary_line[0]*x_end_points + boundary_line[1] # y = mx + c
    if line_count == len(boundary_lines) - 1:
        plt.plot(x_end_points, y_line, color = 'k')
    else:
        plt.plot(x_end_points, y_line, '-.g')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
"""