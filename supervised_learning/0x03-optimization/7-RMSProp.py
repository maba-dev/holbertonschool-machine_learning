#!/usr/bin/env python3
"""updates a variable using the RMSProp optimization algorithm"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm"""
    s = beta2 * s + (1 - beta2) * grad ** 2
    result = var - (alpha * grad / (np.sqrt(s) + epsilon))
    return result, s
