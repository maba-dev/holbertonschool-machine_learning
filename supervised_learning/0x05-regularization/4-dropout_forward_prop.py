#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    outputs = {}
    A = X
    for i in range(L):
        if i > 0:
            outputs['D' + str(i)] = (np.random.rand(
                A.shape[0], A.shape[1]) < keep_prob).astype(int)
            A = np.multiply(A, outputs['D' + str(i)])
            A = A / keep_prob
        outputs["A" + str(i)] = A
        Z = np.dot(weights["W" + str(i + 1)], A) + weights["b" + str(i + 1)]
        if i == L - 1:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            outputs["A" + str(i + 1)] = A
        else:
            A = np.sinh(Z) / np.cosh(Z)
    return (outputs)
