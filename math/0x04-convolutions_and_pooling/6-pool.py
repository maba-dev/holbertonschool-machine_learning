#!/usr/bin/env python3
""" performs pooling on images"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int((h - kh) / sh) + 1
    ow = int((w - kw) / sw) + 1

    pooled_images = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            window = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled_window = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled_window = np.mean(window, axis=(1, 2))
            pooled_images[:, i, j, :] = pooled_window
    return pooled_images
