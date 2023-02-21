#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h, output_w = h - kh + 1, w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    return output
