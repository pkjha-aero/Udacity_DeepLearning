#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pkjha
"""
import numpy as np
import matplotlib.pyplot as plt

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    x = np.array(L)
    x_exp = np.exp(x)
    denominator = np.sum(x_exp)
    x_softmax = x_exp/denominator
    softmax_list = list(x_softmax)
    
    return softmax_list

#L = [5,6,7]
L = range(-10, 20)
L_softmax = softmax(L)

plt.figure(1)
plt.plot(L, L_softmax, color = 'b')

#plt.xlim([-0.25, 1.25])
#plt.ylim([-0.25, 1.25])