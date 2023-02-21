#!/usr/bin/env python3
"""that performs a convolution on grayscale images """


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ that performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Padding
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=0)

    # Convolution
    h_out = int((h + 2*ph - kh) / sh) + 1
    w_out = int((w + 2*pw - kw) / sw) + 1
    output = np.zeros((m, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            output[:, i, j] = np.sum(
                images_padded[:, i*sh:i*sh+kh,
                              j*sw:j*sw+kw] * kernel, axis=(1, 2))
    return output
