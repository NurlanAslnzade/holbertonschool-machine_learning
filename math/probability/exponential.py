#!/usr/bin/env python3
"""
Script to calculate a Exponential distribution
"""


class Exponential:
    """
    class to call the methods of exponential distribution
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        init
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        calculating pdf of exponential distribution
        """
        if x < 0:
            return 0
        pdf = self.lambtha * Exponential.e**((-self.lambtha) * x)
        return pdf

    def cdf(self, x):
        """
        calculating cdf of exponential distribution
        """
        if x < 0:
            return 0
        cdf = 1 - Exponential.e**((-self.lambtha) * x)
        return cdf
