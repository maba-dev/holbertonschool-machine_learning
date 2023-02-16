#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using keras """


import tensorflow.keras as K


def lenet5(X):
    """builds a modified version of the LeNet-5 architecture using keras """
    initializer = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(6, (5, 5), padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(16,
                            (5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(120, activation='relu',
                            kernel_initializer=initializer)(flatten)
    dense2 = K.layers.Dense(84, activation='relu',
                            kernel_initializer=initializer)(dense1)
    output = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=initializer)(dense2)
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
