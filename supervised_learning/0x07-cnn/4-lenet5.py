#!/usr/bin/env python3
""" builds a modified version of the LeNet-5 architecture using tensorflow"""


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """build a modified version of the LeNet-5 architecture using tensorflow"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    layer1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same', activation='relu',
                              kernel_initializer=init)(x)
    layer2 = tf.layers.MaxPooling2D((2, 2), strides=(2, 2))(layer1)
    layer3 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                              padding='valid', activation='relu',
                              kernel_initializer=init)(layer2)
    layer4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer3)
    flat = tf.layers.Flatten()(layer4)
    layer5 = tf.layers.Dense(120, activation='relu',
                             kernel_initializer=init)(flat)
    layer6 = tf.layers.Dense(84, activation='relu',
                             kernel_initializer=init)(layer5)
    logits = tf.layers.Dense(10, kernel_initializer=init)(layer6)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    grady = tf.train.AdamOptimizer()
    op = grady.minimize(loss)
    acc = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32), name="Mean")
    y_pred = tf.nn.softmax(logits)
    return y_pred, op, loss, accuracy
