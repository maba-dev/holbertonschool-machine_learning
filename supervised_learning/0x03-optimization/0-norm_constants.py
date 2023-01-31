#!/usr/bin/env python3
""" calculates the normalization (standardization) constants of a matrix"""


import numpy as np


def normalization_constants(X):
    """define the function """
    mean = (1 / len(X)) * sum(X)
    variance = (1 / len(X)) * sum((X - mean) ** 2)
    stdv = np.sqrt(variance)
    return (mean, stdv)
