#!/usr/bin/env python3


def matrix_shape(matrix):
    """ a function that calculates the shape of a matrix: """


    if type(matrix) is not list:
        return None
    shape = []
    while (type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
