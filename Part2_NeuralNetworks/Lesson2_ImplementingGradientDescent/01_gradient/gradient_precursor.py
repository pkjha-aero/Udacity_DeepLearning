#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 08:25:14 2021

@author: pkjha
"""

import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consolidated, so there are
###       fewer variable names than in the above sample code

# Calculate the node's linear combination of inputs and weights
h = np.dot(w, x)

# Calculate output of neural network
y_hat = sigmoid(h)

# Calculate error of neural network
error = y - y_hat

# Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.#
#delta = error                    # This is for cross-entropy
delta = error * sigmoid_prime(h) # This is for sum of squarred error (SSE)

# Calculate change in weights
del_w = learnrate * delta * x

print('Neural Network output:')
print(y_hat)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)