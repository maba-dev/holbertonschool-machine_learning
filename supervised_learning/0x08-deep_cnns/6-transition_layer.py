#!/usr/bin/env python3
"""
    builds a transition layer as described
    in Densely Connected Convolutional Networks:
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
        builds a transition layer as described in
        Densely Connected Convolutional Networks:
    """
    conv = K.layers.BatchNormalization()(X)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.Conv2D(int(nb_filters * compression), kernel_size=1,
                           kernel_initializer='he_normal')(conv)
    avg_pool = K.layers.AveragePooling2D(2, strides=2, padding='same')(conv)
    return avg_pool, int(nb_filters * compression)
