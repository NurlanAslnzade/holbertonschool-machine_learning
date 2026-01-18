#!/usr/bin/env python3
"""That calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """
    Return a new list of coefficients representing the derivative
    of the polynomial.
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    for i in poly:
        if not isinstance(i, (int, float)):
            return None

    a = []
    for i in range(1, len(poly)):
        a.append(poly[i] * i)

    if len(a) == 0 or all(i == 0 for i in a):
        return [0]

    return a
