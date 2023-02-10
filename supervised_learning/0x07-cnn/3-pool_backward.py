#!/usr/bin/env python3
"""performs back propagation over a pooling layer of a neural network"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer of a neural network"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    kh, kw = kernel_shape
    for i in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for c in range(c_new):
                    ys = stride[1] * y
                    xs = stride[0] * x
                    if mode == 'max':
                        A = A_prev[i, xs:xs + kh, ys:ys + kw, c]
                        mask = A == np.max(A)
                        dA_prev[i, xs:xs + kh,
                                ys:ys + kw, c] += mask * dA[i, x, y, c]
                    elif mode == 'avg':
                        avg = dA[i, x, y, c] / (kh * kw)
                        avg = avg * np.ones(kernel_shape)
                        dA_prev[i, xs:xs + kh, ys:ys + kw, c] += avg
    return dA_prev
