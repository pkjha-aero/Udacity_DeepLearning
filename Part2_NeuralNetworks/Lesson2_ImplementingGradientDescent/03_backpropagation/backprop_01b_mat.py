#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 03 09:56:09 2022

@author: pkjha
"""

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([[0.5, 0.1, -0.2]])
target = 0.6 # y_k = y
learnrate = 0.5 # eta

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([[0.1], 
                                  [-0.3]])

## Forward pass
hidden_layer_input = np.matmul(x, weights_input_hidden) # [h_j] = x_i*W_ij, 1X2 matrix
hidden_layer_output = sigmoid(hidden_layer_input) # a_j = f(h_j), 1X2 matrix

output_layer_in = np.matmul(hidden_layer_output, weights_hidden_output) # o_k = a_j*W_jk, 1X1 matrix
output = sigmoid(output_layer_in) # y_k_hat = f(o_k), 1X1 matrix

## Backwards pass
## Calculate output error
error = (target - output)

# Calculate error term for output layer
# (y - y_k_hat)*sigmoid_prime(a_k) = (y - y_k_hat)*output*(1- output)
output_error_term = error * output * (1 - output)

# Calculate error term for hidden layer
hidden_error_term = np.matmul(output_error_term, weights_hidden_output.T)
hidden_error_term *= hidden_layer_output* (1 - hidden_layer_output)

# Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * np.matmul (hidden_layer_output.T, output_error_term)

# Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * np.matmul(x.T, hidden_error_term)

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
