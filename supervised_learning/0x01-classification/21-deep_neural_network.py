#!/usr/bin/env python3
"""
    Class that defines a deep neural network
    performing binary classification
"""


import numpy as np


class DeepNeuralNetwork():
    """Defines the DeepNeuralNetwork class"""

    def __init__(self, nx, layers):
        """constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')
        inputs = nx
        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        for idx, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(idx)] = np.zeros((layer, 1))
            weights["W{}".format(idx)] = (
                np.random.randn(layer, inputs) * np.sqrt(2 / inputs))
            inputs = layer
        self.__weights = weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def sigmoid(self, X=None, w=None, b=None, z=None):
        """sigmoid activation fuction"""
        if z is None:
            z = np.dot(w, X)
            z = np.add(z, b)
        return (1 / (1 + np.exp(-z)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        i = 0
        n = self.__L
        self.__cache['A0'] = X
        for i in range(1, n+1):
            self.__cache['A'+str(i)] = self.sigmoid(self.__cache['A'+str(i-1)],
                                                    self.weights['W'+str(i)],
                                                    self.weights['b'+str(i)])
        return self.__cache['A'+str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model with logistic regression"""
        m = A.shape[1]
        error = (-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        return 1 / m * np.sum(error)

    def evaluate(self, X, Y):
        """Neurons prediction evaluation"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.where(A > 0.5, 1, 0), cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dz = self.__cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            a_i = 'A' + str(i - 1)
            w_i = 'W' + str(i)
            b_i = 'b' + str(i)
            A = self.__cache[a_i]
            dw = (1 / m) * np.dot(dz, np.transpose(A))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(np.transpose(self.__weights[w_i]), dz) * A * (1 - A)
            self.__weights[w_i] = self.__weights[w_i] - (alpha * dw)
            self.__weights[b_i] = self.__weights[b_i] - (alpha * db)
