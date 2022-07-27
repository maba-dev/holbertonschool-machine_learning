#!/usr/bin/env python3
""" a function that adds two arrays element-wise"""

from operator import le
from unittest import result


def add_arrays(arr1, arr2):
    """adds two arrays element-wise """
    if (len(arr1) != len(arr2)):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
