#!/usr/bin/env python3
""" a function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """ derivative of a polynomial"""
    if len(poly) == 0 or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    result = []
    for i in range(1, len(poly)):
        if type(poly[i]) is not int:
            return None
        result.append(poly[i]*i)
    return result
