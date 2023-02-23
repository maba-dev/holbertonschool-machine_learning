#!/usr/bin/env python3
"""
    that builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
        that builds an identity block as described in
        Deep Residual Learning for Image Recognition (2015)
    """
    conv = K.layers.Conv2D(filters[0], 1,
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
    conv = K.layers.add([conv, A_prev])
    return K.layers.Activation('relu')(conv)
