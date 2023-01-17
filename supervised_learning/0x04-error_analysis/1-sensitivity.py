#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""


import numpy as np


def sensitivity(confusion):
    """ define the function"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    TPR = TP / (TP + FN)
    return TPR
