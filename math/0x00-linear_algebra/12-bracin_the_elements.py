#!/usr/bin/env python3
"""
    a function that performs element-wise addition,
    subtraction, multiplication, and division:
"""


def np_elementwise(mat1, mat2):
    """ calcul matriciel """
    result = []
    result.append(mat1 + mat2)
    result.append(mat1 - mat2)
    result.append(mat1 * mat2)
    result.append(mat1 / mat2)
    return result
