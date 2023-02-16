#!/usr/bin/env python3
"""  trains a model using mini-batch gradient descent"""


import numpy as np
from tensorflow import keras


def train_model(network,
                data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent:"""
    model = keras.models.clone_model(network)
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    num_samples = data.shape[0]
    steps_per_epoch = num_samples // batch_size
    history = model.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle
    )
    return history
