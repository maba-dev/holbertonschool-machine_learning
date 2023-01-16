#!/usr/bin/env python3
"""the function create layer """


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """creates layer for NN in Tensorflow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=init)
