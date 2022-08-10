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
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            _data = []
            N = 1 / len(data)
            for i in data:
                _data.append((i - self.mean) ** 2)
            self.stddev = (N * sum(_data)) ** (1 / 2)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-values"""
        pi = 3.1415926536
        e = 2.7182818285
        exp = (- (1 / 2) * (((x - self.mean) / self.stddev) ** 2))
        return (1 / (self.stddev * ((2 * pi) ** (1/2))) * (e ** exp))
