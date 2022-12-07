#!/usr/bin/env python3
"""a  that calculates sum_{i=1}^{n} i^2:"""


def summation_i_squared(n):
    """SUMS of SQUARES with consecutive numbers """
    if type(n) is int and n > 0:
        return int(((n * (n + 1) * (2 * n + 1)) / 6))
