#!/usr/bin/env python3
"""
    creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm:
"""


import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        creates the training operation for a neural network in
        tensorflow using the RMSProp optimization algorithm:
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
