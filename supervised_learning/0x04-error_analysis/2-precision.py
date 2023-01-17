#!/usr/bin/env python3
"""that calculates the precision for each class in a confusion matrix """


import numpy as np


def precision(confusion):
    """define the function """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    PPV = TP / (TP + FP)
    return PPV
