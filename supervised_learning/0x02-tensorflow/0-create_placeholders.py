#!/usr/bin/env python3
"""Placehoders """

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ the function that returns two placeholders, x and y """
    x = tf.placeholder(name='x', shape=(None, nx), dtype=tf.float32)
    y = tf.placeholder(name='y', shape=(None, classes), dtype=tf.float32)
    return (x, y)
