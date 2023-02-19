#!/usr/bin/env python3
"""builds an inception block as described in Going Deeper with Convolutions"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """build an inception block as described in Going Deep with Convolution"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3), padding='same', activation='relu')(conv3x3)
    conv5x5 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5), padding='same', activation='relu')(conv5x5)
    maxpool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    maxpool = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1), padding='same', activation='relu')(maxpool)
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, maxpool])
    return output
