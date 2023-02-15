#!/usr/bin/env python3
"""builds a neural network with the Keras library """


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    x_input = Input(shape=(nx,))

    prev_layer = x_input
    for i in range(len(layers)):
        layer_size = layers[i]
        activation = activations[i]
        layer = Dense(
            layer_size, activation=activation,
            kernel_regularizer=regularizers.l2(lambtha))(prev_layer)
        if keep_prob < 1.0:
            layer = Dropout(1 - keep_prob)(layer)
        prev_layer = layer
    model = Model(inputs=x_input, outputs=prev_layer)
    return model
