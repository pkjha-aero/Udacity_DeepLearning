#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 01:40:23 2021

@author: pkjha
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

#       We haven't provided the sigmoid_prime function like we did in
#       the previous lesson to encourage you to come up with a more
#       efficient solution. If you need a hint, check out the comments
#       in solution.py from the previous lecture.

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

loss_arr = []
for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features, targets):
        # Loop through all records, x is the input, y is the target

        # Note: We haven't included the h variable from the previous
        #       lesson. You can add it if you want, or you can calculate
        #       the h together with the output

        # Calculate the output
        h = np.dot(x, weights)
        y_hat = sigmoid(h) # y_hat

        # Calculate the error
        error = y - y_hat

        # Calculate the error term (SSE)
        # (y - y_hat)*sigmoid_prime(y_hat) = (y - y_hat)*sigmoid(y_hat)*(1- sigmoid(y_hat))
        delta = error * y_hat * (1 - y_hat)


        # Calculate the change in weights for this sample
        #       and add it to the total weight change
        del_w +=  delta * x

    # Update weights using the learning rate and the average change in weights
    weights += learnrate * del_w /n_records

    out = sigmoid(np.dot(features, weights))
    loss = np.mean((out - targets) ** 2)
    loss_arr.append(loss)
    
    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        if last_loss and last_loss < loss:
            print("Epoch:", e, "Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Epoch:", e, "Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

# Plotting the error
plt.title("Error Plot")
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.plot(loss_arr)
plt.show()
