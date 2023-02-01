#!/usr/bin/env python3
""" """

import tensorflow.compat.v1 as tf
""" """


def create_batch_norm_layer(prev, n, activation):
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, activation=None, kernel_initializer=init)
    x = dense(prev)
    mean, variance = tf.nn.moments(x, axes=[0])
    beta = tf.Variable(tf.zeros([1, n]), dtype=tf.float32)
    gamma = tf.Variable(tf.ones([1, n]), dtype=tf.float32)
    epsilon = 1e-8
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    return activation(x)
