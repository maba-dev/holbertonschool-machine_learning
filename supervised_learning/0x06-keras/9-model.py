#!/usr/bin/env python3
"""saves a model’s weights and loads a model’s weights"""


import tensorflow.keras as K


def save_model(network, filename):
    """ saves a model’s weights"""
    network.save(filename)
    return None


def load_model(filename):
    """ loads a model’s weights: """
    return K.models.load_model(filename)
