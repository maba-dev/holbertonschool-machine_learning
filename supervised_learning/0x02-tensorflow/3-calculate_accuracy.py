#!/usr/bin/env python3
""" Calculates accuracy of prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """define the function"""
    pred = tf.math.argmax(y_pred, axis=1)
    ny = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, ny)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
