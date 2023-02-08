#!/usr/bin/env python3
""" performs forward propagation over a convolutional layer of a neural network:"""


import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer of a neural network:"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    
    if padding == "same":
        ph = int((h_prev - 1) * sh + kh - h_prev), 0
        pw = int((w_prev - 1) * sw + kw - w_prev), 0
        A_prev = np.pad(A_prev, ((0, 0), (ph//2, ph//2), (pw//2, pw//2), (0, 0)), "constant")
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    
    h = int(1 + (h_prev - kh) / sh)
    w = int(1 + (w_prev - kw) / sw)
    conv = np.zeros((m, h, w, c_new))
    
    for i in range(h):
        for j in range(w):
            x = i * sh
            y = j * sw
            a_slice = A_prev[:, x:x+kh, y:y+kw, :]
            for k in range(c_new):
                conv[:, i, j, k] = np.sum(a_slice * W[:, :, :, k], axis=(1, 2, 3))
                
    Z = conv + b
    return Z
