#!/usr/bin/env python3
""" a function that concatenates two matrices along a specific axis"""


def cat_arrays(arr1, arr2):
    """concatenates two arrays """
    if type(arr1) is not list or type(arr2) is not list:
        return None
    result = []
    j = 0
    for i in range(0, len(arr1) + len(arr2)):
        if i < len(arr1):
            result.append(arr1[i])
        else:
            result.append(arr2[j])
            j = j + 1
    return result


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis """
    result = []
    if axis == 0:
        if (len(mat1[0]) != len(mat2[0])):
            return None
        for i in mat1:
            result.append(list(i))
        for i in mat2:
            result.append(list(i))
        return result
    if (len(mat1) != len(mat2)):
        return None
    for i in range(len(mat1)):
        result.append(list())
        result[i] = cat_arrays(mat1[i], mat2[i])
    return result
