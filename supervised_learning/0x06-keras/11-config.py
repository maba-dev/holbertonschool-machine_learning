#!/usr/bin/env python3
"""Saves and Loads a model's configuration in JSON format """


import json
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format."""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """Loads a model with a specific configuration."""
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)
    return network
