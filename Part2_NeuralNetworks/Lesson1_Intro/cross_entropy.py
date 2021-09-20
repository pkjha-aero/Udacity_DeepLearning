#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 01:45:18 2021

@author: pkjha
"""

import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # CE = 0.0
    # for yi, pi in zip(Y, P):
    #     CE -= yi*np.log(pi) + (1.0 - yi)*np.log(1.0 - pi)
    # return CE

    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

Y = [1, 1, 0]
#Y = [0, 0, 1]
P = [0.8, 0.7, 0.1]

CE = cross_entropy(Y, P)
print('Cross-Entropy: {}'.format('%.6f'%CE))