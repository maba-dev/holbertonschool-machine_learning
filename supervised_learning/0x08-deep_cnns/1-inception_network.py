#!/usr/bin/env python3
"""
    builds the inception network as described
    in Going Deeper with Convolutions (2014):
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
        builds the inception network as described
        in Going Deeper with Convolutions (2014):
    """
    x = K.Input(shape=(224, 224, 3))
    conv = K.layers.Conv2D(64, kernel_size=(7, 7),
                           strides=(2, 2),
                           activation="relu", padding='same')(x)
    mpool = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2), padding='same')(conv)
    conv = K.layers.Conv2D(64,
                           kernel_size=(1, 1), strides=(1, 1),
                           activation="relu", padding='same')(mpool)
    conv = K.layers.Conv2D(192, kernel_size=(3, 3),
                           strides=(1, 1),
                           activation="relu", padding='same')(conv)
    mpool = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(conv)
    one = inception_block(mpool, [64, 96, 128, 16, 32, 32])
    two = inception_block(one, [128, 128, 192, 32, 96, 64])
    mpool = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2), padding='same')(two)
    Xa = inception_block(mpool, [192, 96, 208, 16, 48, 64])
    Xb = inception_block(Xa, [160, 112, 224, 24, 64, 64])
    Xc = inception_block(Xb, [128, 128, 256, 24, 64, 64])
    Xd = inception_block(Xc, [112, 144, 288, 32, 64, 64])
    Xe = inception_block(Xd, [256, 160, 320, 32, 128, 128])
    mpool = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(Xe)
    out1 = inception_block(mpool, [256, 160, 320, 32, 128, 128])
    out2 = inception_block(out1, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1), padding='valid')(out2)
    drop = K.layers.Dropout(rate=0.4)(avgpool)
    dense = K.layers.Dense(1000, activation='softmax')(drop)
    model = K.Model(inputs=x, outputs=dense)
    return model
