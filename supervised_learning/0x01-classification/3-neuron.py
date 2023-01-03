#!/usr/bin/env python3
"""
    a class Neuron that defines a single neuron
    performing binary classification
"""


import numpy as np


class Neuron():
    """ define the class"""

    def __init__(self, nx):
        """ class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.b
        self.__A = (1 / (1 + np.exp(-z)))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        c = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))
            )
        return c
