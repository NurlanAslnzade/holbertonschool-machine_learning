#!/usr/bin/env python3
import numpy as np
"""asdasda"""


def np_shape(matrix):
    """Murad necesen"""
    return tuple(len(matrix), np_shape(matrix[0]) if matrix else ())

