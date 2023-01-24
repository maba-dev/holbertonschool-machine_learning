#!/usr/bin/env python3
""" creates the training operation for the network"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """define the function"""
    operation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return operation
