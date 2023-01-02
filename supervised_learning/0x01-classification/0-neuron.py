#!/usr/bin/env python3
import numpy as np
"""
    a class Neuron that defines a single neuron
    performing binary classification
"""


class Neuron():
    """ define the class"""

    def __init__(self, nx):
        """ class constructor"""

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.rand(1, nx)
        self.b = 0
        self.A = 0
