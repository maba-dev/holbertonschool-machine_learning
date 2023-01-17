#!/usr/bin/env python3
""" that calculates the specificity for each class in a confusion matrix"""


import numpy as np


def specificity(confusion):
    """ define the function"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    TNR = TN / (TN + FP)
    return TNR
