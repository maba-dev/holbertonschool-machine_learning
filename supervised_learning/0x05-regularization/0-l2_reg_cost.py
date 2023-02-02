#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization: """


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization:"""
    reg_cost = 0
    for i in range(1, L):
        reg_cost += np.sum(np.square(weights["W" + str(i)]))
    reg_cost = (lambtha / (2 * m)) * reg_cost
    return cost + reg_cost
