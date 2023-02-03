#!/usr/bin/env python3
"""creates a layer of a neural network using dropout: """


import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates a layer of a neural network using dropout:"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    L2 = tf.layers.Dropout(keep_prob)
    dropout = tf.layers.Dense(
        n, activation, name='layer',
        kernel_initializer=init, kernel_regularizer=L2)
    return dropout(prev)
