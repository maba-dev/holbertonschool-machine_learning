#!/usr/bin/env python3
""" save the best iteration of the model:"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """ save the best iteration of the model:"""
    def learning_rate_decay(epoch):
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               mode='min')
        callbacks.append(early_stop)
        learning = K.callbacks.LearningRateScheduler(
            schedule=learning_rate_decay, verbose=1)
        callbacks.append(learning)
    if validation_data and save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                     save_best_only=True))
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
