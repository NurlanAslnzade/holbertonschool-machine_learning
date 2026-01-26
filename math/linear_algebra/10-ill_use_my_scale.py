#!/usr/bin/env python3
"""asdasda"""


def np_shape(matrix):
    """Murad necesen"""
    return tuple(len(matrix), np_shape(matrix[0]) if matrix else ())

