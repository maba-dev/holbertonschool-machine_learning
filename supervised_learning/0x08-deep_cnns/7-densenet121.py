#!/usr/bin/env python3
"""
    builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks:

"""


import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        builds the DenseNet-121 architecture as described
        in Densely Connected Convolutional Networks:
    """
    data = K.Input((224, 224, 3))
    conv = K.layers.BatchNormalization()(data)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.Conv2D(growth_rate * 2, 7, 2,
                           padding='same',
                           kernel_initializer='he_normal')(conv)
    conv = K.layers.MaxPool2D(2)(conv)
    conv, nb_filters = dense_block(conv, growth_rate * 2, growth_rate, 6)
    conv, nb_filters = transition_layer(conv, nb_filters, compression)
    conv, nb_filters = dense_block(conv, nb_filters, growth_rate, 12)
    conv, nb_filters = transition_layer(conv, nb_filters, compression)
    conv, nb_filters = dense_block(conv, nb_filters, growth_rate, 24)
    conv, nb_filters = transition_layer(conv, nb_filters, compression)
    conv, nb_filters = dense_block(conv, nb_filters, growth_rate, 16)
    conv = K.layers.AvgPool2D(7)(conv)
    conv = K.layers.Dense(1000, kernel_initializer='he_normal',
                          activation='softmax')(conv)
    return K.Model(data, conv)
