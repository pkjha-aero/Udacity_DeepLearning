#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 01:40:28 2021

@author: pkjha
"""

import numpy as np
import pandas as pd
import os.path as pth
#admissions = pd.read_csv('binary.csv')
admissions = pd.read_csv(pth.join(pth.pardir, 'binary.csv'))

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.loc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1).values.astype(float), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1).values.astype(float), test_data['admit']