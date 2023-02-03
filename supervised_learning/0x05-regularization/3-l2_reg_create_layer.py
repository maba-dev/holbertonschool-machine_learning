#!/usr/bin/env python3
""" creates a tensorflow layer that includes L2 regularization: """


import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer that includes L2 regularization: """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    L2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(
        n, activation, name='layer',
        kernel_initializer=init, kernel_regularizer=L2)
    return layer(prev)
