#!/usr/bin/env python3
"""a class Binomial that represents a binomial distribution """


from math import factorial


class Binomial:
    """a class Binomial distribution  """
    def __init__(self, data=None, n=1, p=0.5):
        """constructor"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            N = 1 / len(data)
            _data = []
            for i in data:
                _data.append((i - mean) ** 2)
            variance = sum(_data) * N
            self.p = 1 - (variance) / (mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial_1 = 1
        for i in range(1, self.n + 1):
            factorial_1 = factorial_1 * i
        factorial_2 = 1
        for i in range(1, k + 1):
            factorial_2 = factorial_2 * i
        factorial_3 = 1
        for i in range(1, self.n - k + 1):
            factorial_3 = factorial_3 * i
        comb = (factorial_1 / (factorial_2 * factorial_3))
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
