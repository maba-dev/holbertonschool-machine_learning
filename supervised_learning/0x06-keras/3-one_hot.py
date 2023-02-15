#!/usr/bin/env python3
"""converts a label vector into a one-hot matrix:"""


import numpy as np


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix:"""
    if classes is None:
        classes = np.max(labels) + 1
    one_hot_matrix = np.eye(classes)[labels.reshape(-1)]
    return one_hot_matrix
