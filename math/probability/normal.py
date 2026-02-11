#!/usr/bin/env python3
"""
initializing normal dist class
"""


class Normal:
    """
    class to show functions of Normal distribution
    """
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data: list of data given
        mean: self attribute of the mean of the data
        stddev: standard error of the data
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sigma = 0
            for i in range(len(data)):
                sigma += (data[i] - self.mean)**2
            self.stddev = (sigma / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        calculating z-score
        """
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """
        calculating x_value
        """
        x = z * self.stddev + self.mean
        return x

    def pdf(self, x):
        """
        calculating pdf of normal distribution
        """
        coeff = 1 / (((2 * Normal.π) ** 0.5) * self.stddev)
        exponent = -0.5 * (self.z_score(x) ** 2)
        f_x = coeff * (Normal.e ** exponent)
        return f_x

    def cdf(self, x):
        """
        calculating cdf of normal distribution
        """
        xa = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        errof = ((4 / Normal.π) ** 0.5) * (
            xa
            - (xa ** 3) / 3
            + (xa ** 5) / 10
            - (xa ** 7) / 42
            + (xa ** 9) / 216
        )
        cdf = (1 + errof) / 2
        return cdf
