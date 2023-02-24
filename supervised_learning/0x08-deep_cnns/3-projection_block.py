#!/usr/bin/env python3
"""
    builds a projection block as described in Deep
    Residual Learning for Image Recognition (2015):
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
        builds a projection block as described in Deep
        Residual Learning for Image Recognition (2015):
    """
    conv = K.layers.Conv2D(filters[0], 1, s,
                           kernel_initializer='he_normal')(A_prev)
    conv = K.layers.BatchNormalization()(conv)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.Conv2D(filters[1], 3, padding='same',
                           kernel_initializer='he_normal')(conv)
    conv = K.layers.BatchNormalization()(conv)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.Conv2D(filters[2], 1,
                           kernel_initializer='he_normal')(conv)
    conv = K.layers.BatchNormalization()(conv)
    conv2 = K.layers.Conv2D(filters[2], 1, s,
                            kernel_initializer='he_normal')(A_prev)
    conv2 = K.layers.BatchNormalization()(conv2)
    conv = K.layers.add([conv, conv2])
    return K.layers.Activation('relu')(conv)
