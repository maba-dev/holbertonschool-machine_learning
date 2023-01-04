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

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, self.cost(Y, A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dw = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
