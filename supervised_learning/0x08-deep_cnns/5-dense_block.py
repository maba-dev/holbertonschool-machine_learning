#!/usr/bin/env python3
"""
    builds a dense block as described in
    Densely Connected Convolutional Networks
"""


import tensorflow.keras as K


def dense_block(X, nb_filter, growth_rate, layers):
    """
        builds a dense block as described in
        Densely Connected Convolutional Networks
    """
    for i in range(layers):
        conv = K.layers.BatchNormalization()(X)
        conv = K.layers.Activation('relu')(conv)
        conv = K.layers.Conv2D(4 * growth_rate, 1,
                               kernel_initializer='he_normal')(conv)
        conv2 = K.layers.BatchNormalization()(conv)
        conv = K.layers.Activation('relu')(conv2)
        X_conv = K.layers.Conv2D(growth_rate, 3, padding='same',
                                 kernel_initializer='he_normal')(conv)
        X = K.layers.concatenate([X, X_conv])
        nb_filter += growth_rate
    return X, nb_filter
