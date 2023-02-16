#!/usr/bin/env python3
""" Train a deep learning model using mini-batch gradient descent """


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Trains a deep learning model using mini-batch gradient descent."""

    def learning_rate_decay(epoch):
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               mode='min')
        learning = K.callbacks.LearningRateScheduler(
            schedule=learning_rate_decay, verbose=1
        )
        callbacks.append(learning)
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
