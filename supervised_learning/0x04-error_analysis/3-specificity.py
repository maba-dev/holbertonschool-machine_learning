#!/usr/bin/env python3
""" """


import numpy as np


def specificity(confusion):
    """ """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    TNR = TN / (TN + FP)
    return TNR
