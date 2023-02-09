#!/usr/bin/env python3
"""back propagation over a convolutional layer of a neural network"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """back propagation over a convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = max((h_prev - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_h, pad_w = pad_h // 2, pad_w // 2
        A_prev = np.pad(
            A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant')

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    a_slice = a_prev[
                        vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :]+= W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA_prev, dW, db
