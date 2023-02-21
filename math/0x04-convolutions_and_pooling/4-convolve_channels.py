#!/usr/bin/env python3
""" performs a convolution on images with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # Pad the images
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Calculate output shape
    oh = int((h + 2*ph - kh) / sh + 1)
    ow = int((w + 2*pw - kw) / sw + 1)

    # Initialize output tensor
    output = np.zeros((m, oh, ow, 1))

    # Perform convolution
    for i in range(oh):
        for j in range(ow):
            output[:, i, j, 0] = np.sum(
                images_padded[:, i*sh:i*sh+kh,
                              j*sw:j*sw+kw, :] * kernel, axis=(1, 2, 3))
    return output
