#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow.compat.v1 as tf

def create_layer(prev, n, activation):
    """creates layer for NN in Tensorflow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=init)
