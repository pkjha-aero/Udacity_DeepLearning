#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:54:52 2021

@author: pkjha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from GradientDescent import sigmoid
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

"""
Data
"""
p = np.array([0.4, 0.6])
W_array = np.array([[2, 6], [3, 5], [5,4]]).T
b_array = np.array([-2, -2.2, -3])

linear_comb = np.dot(p, W_array) + b_array

p_combined = sigmoid(linear_comb)