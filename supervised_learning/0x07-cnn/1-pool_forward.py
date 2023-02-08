#!/usr/bin/env python3
"""performs forward propagation over a pooling layer of a neural network"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int(1 + (h_prev - kh) / sh)
    w_out = int(1 + (w_prev - kw) / sw)
    pool_out = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            pool_window = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                pool_out[:, i, j, :] = np.max(pool_window, axis=(1, 2))
            elif mode == 'avg':
                pool_out[:, i, j, :] = np.mean(pool_window, axis=(1, 2))

    return pool_out
