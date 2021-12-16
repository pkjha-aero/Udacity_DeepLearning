#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pkjha
"""

import pandas as pd

class AND_OPERATOR:
    weight1 =  1.1
    weight2 =  1.1
    bias    = -2.0
    
    def __init__(self):
        print('AND OPERATOR:-> w1 = {}, w2 = {}, b = {}\n'.format(self.weight1, self.weight2, self.bias))
    
    def operate(self, x1, x2):
        linear_combination = self.weight1 * x1 + self.weight2 * x2 + self.bias
        output = int(linear_combination >= 0)
        return linear_combination, output

class OR_OPERATOR:
    weight1 =  1.1
    weight2 =  1.1
    bias    = -1.0
    
    def __init__(self):
        print('OR OPERATOR:-> w1 = {}, w2 = {}, b = {}\n'.format(self.weight1, self.weight2, self.bias))
    
    def operate(self, x1, x2):
        linear_combination = self.weight1 * x1 + self.weight2 * x2 + self.bias
        output = int(linear_combination >= 0)
        return linear_combination, output
    
class NOT_OPERATOR:
    weight1 = 0.0
    weight2 = -1.0
    bias = 0.5
    
    def __init__(self):
        print('NOT OPERATOR:-> w1 = {}, w2 = {}, b = {}\n'.format(self.weight1, self.weight2, self.bias))
    
    def operate(self, x1, x2):
        linear_combination = self.weight1 * x1 + self.weight2 * x2 + self.bias
        output = int(linear_combination >= 0)
        return linear_combination, output
    def operate_one(self, x):
        return self.operate(0, x)
        
class XOR_OPERATOR:
    def __init__(self):
        print('XOR OPERATOR:-> \n')
        self.and_op = AND_OPERATOR()
        self.or_op = OR_OPERATOR()
        self.not_op = NOT_OPERATOR()
        
        self.A = self.and_op
        self.B = self.or_op
        self.C = self.not_op
        
    def operate(self, x1, x2):
        lc_A, U = self.A.operate(x1, x2)
        lc_B, V = self.B.operate(x1, x2)
        lc_C, W = self.C.operate_one(U)
        
        return self.and_op.operate(W, V)

def test_outputs(operator, test_inputs, correct_outputs):
    outputs = []

    # Generate and check output
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination, output = operator.operate(test_input[0], test_input[1])
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

    # Print output
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
    if not num_wrong:
        print('Nice!  You got it all correct.\n')
    else:
        print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
    print(output_frame.to_string(index=False))
    print('\n')

def test_AND_Operator():
    and_op = AND_OPERATOR()
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, False, False, True]
    test_outputs(and_op, test_inputs, correct_outputs)

def test_OR_Operator():
    or_op = OR_OPERATOR()
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, True, True, True]
    test_outputs(or_op, test_inputs, correct_outputs)
    
def test_NOT_Operator():
    not_op = NOT_OPERATOR()
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [True, False, True, False]
    test_outputs(not_op, test_inputs, correct_outputs)

def test_XOR_Operator():
    xor_op = XOR_OPERATOR()
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = correct_outputs = [False, True, True, False]
    test_outputs(xor_op, test_inputs, correct_outputs)

test_AND_Operator()
test_OR_Operator()
test_NOT_Operator()
test_XOR_Operator()
