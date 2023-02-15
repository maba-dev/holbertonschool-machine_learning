#!/usr/bin/env python3
"""builds a neural network with the Keras library """


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    x_input = K.Input(shape=(nx,))
    layer = K.layers.Dense(
                layers[0], activation=activations[0],
                kernel_regularizer=K.regularizers.l2(lambtha))(x_input)
    for i in range(len(layers)):
        if i > 0:
            layer = K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha))(layer)
        if i < len(layers) - 1:
            layer = K.layers.Dropout(1 - keep_prob)(layer)
    model = K.Model(inputs=x_input, outputs=layer)
    return model
