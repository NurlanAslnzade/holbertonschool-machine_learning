#!/usr/bin/env python3
"""a class Poisson that represents a poisson distribution"""


def fact(a):
    '''
    calculating factorial
    '''
    s = 1
    for i in range(1, a+1):
        s *= i
    return s


class Poisson:
    """
    class to call methods of Poisson distribution
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        data: list of data to estimate distribution
        lambtha: number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """
        calculating pmf of poisson distribution
        """
        k = int(k)
        if k < 0:
            return 0
        p = Poisson.e**(-self.lambtha)*self.lambtha**k/fact(k)
        return p

    def cdf(self, k):
        """
        calculating cdf of poisson distribution
        """
        k = int(k)
        if k < 0:
            return 0
        s = 0
        for i in range(k+1):
            s += Poisson.e**(-self.lambtha)*self.lambtha**i/fact(i)
        return s
