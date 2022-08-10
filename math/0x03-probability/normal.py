#!/usr/bin/env python3
""" a class Normal that represents a normal distribution"""


class Normal:
    """a class normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ constructor """
        self.stddev = float(stddev)
        self.mean = float(mean)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple")
            self.mean = sum(data) / len(data)
            _data = []
            N = 1 / len(data)
            for i in data:
                _data.append((i - self.mean) ** 2)
            self.stddev = (N * sum(_data)) ** (1 / 2)
