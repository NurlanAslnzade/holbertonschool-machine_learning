#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Concatenates two matrices along a specific axis using numpy.
    Returns a new numpy.ndarray.
    """
    return np.matmul(mat1, mat2)
