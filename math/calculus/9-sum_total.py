#!/usr/bin/env python3
"""sum of i^2"""


def summation_i_squared(n):
    """sum  i2"""
    if type(n) is not int:
        return None
    elif n < 1:
        return None
    else:
        return (n * (n + 1) * (2 * n + 1)) / 6
