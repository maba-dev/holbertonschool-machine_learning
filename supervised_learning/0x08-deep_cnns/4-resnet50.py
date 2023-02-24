#!/usr/bin/env python3
"""
    builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
"""


import tensorflow.keras as K


identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
        builds the ResNet-50 architecture as described in
        Deep Residual Learning for Image Recognition (2015)
    """
    data = K.layers.Input((224, 224, 3))
    conv = K.layers.Conv2D(64, 7, 2, padding='same',
                           kernel_initializer='he_normal')(data)
    conv = K.layers.BatchNormalization()(conv)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.MaxPool2D(3, 2, padding='same')(conv)
    filter = [64, 64, 256]
    conv = projection_block(conv, filter, 1)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    filter = [128, 128, 512]
    conv = projection_block(conv, filter, 2)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    filter = [256, 256, 1024]
    conv = projection_block(conv, filter, 2)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    filter = [512, 512, 2048]
    conv = projection_block(conv, filter, 2)
    conv = identity_block(conv, filter)
    conv = identity_block(conv, filter)
    conv = K.layers.AvgPool2D(7)(conv)
    conv = K.layers.Dense(1000,
                          kernel_initializer='he_normal',
                          activation='softmax')(conv)
    model = K.Model(data, conv)
    return model
