#!/usr/bin/env python3
"""trains a model using mini-batch gradient descent"""


import tensorflow.keras as K


def train_model(network,
                data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent:"""

    History = model.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle
    )
    return History
