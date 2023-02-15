#!/usr/bin/env python3
"""builds a neural network with the Keras library """

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    x_input = K.layers.Input(shape=(nx,))

    prev_layer = x_input
    for i in range(len(layers)):
        layer_size = layers[i]
        activation = activations[i]
        layer = K.layers.Dense(
            layer_size, activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha))(prev_layer)
        if keep_prob < 1.0:
            layer = K.layers.Dropout(1 - keep_prob)(layer)
        prev_layer = layer
    model = K.models.Model(inputs=x_input, outputs=prev_layer)
    return model
