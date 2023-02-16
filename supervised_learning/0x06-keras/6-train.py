#!/usr/bin/env python3
"""train the model using early stopping """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """train the model using early stopping """
    callbacks = []
    if validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               mode='min')
        callbacks.append(early_stop)
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
