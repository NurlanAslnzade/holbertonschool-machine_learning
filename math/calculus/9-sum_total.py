#!/usr/bin/env python3
"""sum of i^2"""


def summation_i_squared(n):
    """
    sum of i^2
    """
    if is not isintance(n, int):
        return None
    elif n < 1:
        return None
    else:
        return (n * (n + 1) * (2 * n + 1)) / 6
