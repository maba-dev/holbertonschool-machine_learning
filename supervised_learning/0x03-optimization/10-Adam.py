#!/usr/bin/env python3
""" creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm:
"""


import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates the training operation for a neural network
        in tensorflow using the Adam optimization algorithm:
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
