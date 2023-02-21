#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding """


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output_h, output_w = h + 2 * pad_h - kh + 1, w + 2 * pad_w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    return output
