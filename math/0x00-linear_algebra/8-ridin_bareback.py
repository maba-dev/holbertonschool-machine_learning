#!/usr/bin/env python3
"""  a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """ performs matrix multiplication"""
    if (len(mat1[0]) != len(mat2)):
        return None
    result = []
    for i in range(len(mat1)):
        line_matrix = []
        for j in range(len(mat2[0])):
            som = 0
            for k in range(len(mat1[0])):
                som += mat1[i][k] * mat2[k][j]
            line_matrix.append(som)
        result.append(line_matrix)
    return result
