#!/usr/bin/env python3
""" performs forward propagation over a convolutional layer of a neural network:"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer of a neural network:"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    
    if padding == "same":
        ph = (h_prev - 1) * sh + kh - h_prev
        pw = (w_prev - 1) * sw + kw - w_prev
        A_prev = np.pad(A_prev, ((0,0), (ph//2, ph//2), (pw//2, pw//2), (0,0)), 'constant')
    else:
        ph, pw = 0, 0
    
    h = int((h_prev + 2 * ph - kh) / sh + 1)
    w = int((w_prev + 2 * pw - kw) / sw + 1)
    
    Z = np.zeros((m, h, w, c_new))
    for i in range(h):
        for j in range(w):
            for k in range(c_new):
                start_h = i * sh
                end_h = start_h + kh
                start_w = j * sw
                end_w = start_w + kw
                Z[:, i, j, k] = np.sum(A_prev[:, start_h:end_h, start_w:end_w, :] * W[:,:,:,k], axis=(1,2,3)) + b[0,0,0,k]
    
    A = activation(Z)
    cache = (A_prev, W, b, stride, padding)
    
    return A, cache
