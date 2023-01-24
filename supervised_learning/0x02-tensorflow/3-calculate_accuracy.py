#!/usr/bin/env python3
"""the function  that Calculates accuracy of prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """define the function"""
    correct_preduction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.math.reduce_mean(tf.cast(correct_preduction, tf.float32))
    return accuracy
