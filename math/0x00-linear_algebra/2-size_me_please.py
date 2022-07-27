#!/usr/bin/env python3
""" a function that calculates the shape of a matrix """


def matrix_shape(matrix):
    """ the shape of a matrix """

    shape = []
    while (type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
