#!/usr/bin/env python3
"""performs a same convolution on grayscale images """


import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h, output_w = h, w
    pad_h = int((kh - 1) / 2)
    pad_w = int((kw - 1) / 2)
    padded_images = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    return output
