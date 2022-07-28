#!/usr/bin/env python3
""" a function that concatenates two arrays:"""


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
