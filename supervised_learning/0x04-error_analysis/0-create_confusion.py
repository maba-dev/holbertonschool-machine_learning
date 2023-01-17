#!/usr/bin/env python3
"""the function that creates a confusion matrix: """


import numpy as np


def create_confusion_matrix(labels, logits):
    """ define the function"""
    return np.matmul(labels.T, logits)
