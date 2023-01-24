#!/usr/bin/env python3
"""calculates the softmax cross-entropy loss of a prediction"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """define the function """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
