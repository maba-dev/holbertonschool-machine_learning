#!/usr/bin/env python3
"""a function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """calculates the integral of a polynomia """
    if type(poly) is not list:
        return None
    if not poly:
        return None
    if len(poly) == 0:
        return None
    if type(C) is not int:
        return None
    if poly == [0]:
        return [C]
    result = []
    result.append(C)
    for i in range(0, len(poly)):
        if type(poly[i]) is not int:
            return None
        coef = poly[i] / (i + 1)
        if coef.is_integer():
            result.append(int(coef))
        else:
            result.append(coef)
    return result
