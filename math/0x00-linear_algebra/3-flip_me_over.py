#!/usr/bin/env python3
""" a function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """ the transpose of a 2D matrix"""
    t_matrix = []
    for j in range(len(matrix[0])):
        line_matrix = []
        for i in matrix:
            line_matrix.append(i[j])
        t_matrix.append(line_matrix)
    return t_matrix
