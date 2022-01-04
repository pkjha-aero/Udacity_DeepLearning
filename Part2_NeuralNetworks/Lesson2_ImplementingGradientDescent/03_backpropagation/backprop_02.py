#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 03 17:56:09 2022

@author: pkjha
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden)) # N_I X N_H matrix
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden) # N_H X N_O matrix

loss_arr = []

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # Calculate the output
        hidden_input = np.dot(x, weights_input_hidden) # [h_j] = x_i*W_ij, 1 X N_H matrix
        hidden_output = sigmoid(hidden_input) # a_j = f(h_j), 1 X N_H matrix
        
        output_layer_in = np.dot(hidden_output, weights_hidden_output) # o_k = a_j*W_jk, 1 X N_O matrix
        output = sigmoid(output_layer_in) # y_k_hat = f(o_k), 1 X N_O matrix

        ## Backward pass ##
        # Calculate the network's prediction error
        error = y - output

        # Calculate error term for the output unit
        # (y - y_k_hat)*sigmoid_prime(y_k_hat) = (y - y_k_hat)*output*(1- output)
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)
        
        # Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        
        # Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x [:, None]

    # Update weights  (don't forget to division by n_records or number of samples)
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    hidden_output = sigmoid(np.dot(x, weights_input_hidden))
    out = sigmoid(np.dot(hidden_output,
                         weights_hidden_output))
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
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

# Plotting the error
plt.title("Error Plot")
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.plot(loss_arr)
plt.show()
