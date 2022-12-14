#!/usr/bin/env python3
""" a class Poisson that represents a poisson distribution """


class Poisson:
    """ a class poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """ constructor"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factoriel_k = 1
        e = 2.7182818285
        for i in range(1, k + 1):
            factoriel_k = factoriel_k * i
        return ((self.lambtha ** k) * (e ** (-(self.lambtha))) / factoriel_k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        CDF = 0
        for i in range(0, k + 1):
            CDF = CDF + self.pmf(i)
        return CDF
