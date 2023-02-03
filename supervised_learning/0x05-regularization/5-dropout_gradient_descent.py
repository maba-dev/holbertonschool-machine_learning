#!/usr/bin/env python3
""" updates the weights of a neural network with
    Dropout regularization using gradient descent
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        updates the weights of a neural network with
        Dropout regularization using gradient descent:
    """
    m = Y.shape[1]
    dz2 = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dz1 = (W.T @ dz2) * (1 - (A * A))
        if i > 1:
            dz1 *= cache["D" + str(i - 1)] / keep_prob
        dw = dz2 @ A.T / m
        db = np.sum(dz2, axis=1, keepdims=True) / m
        dz2 = dz1
        weights["W" + str(i)] -= alpha * dw
        weights["b" + str(i)] -= alpha * db
