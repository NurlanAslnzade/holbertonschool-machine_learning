#!/usr/bin/env python3
def np_shape(matrix):
    dims = []
    while matrix.ndim > 0:
        dims.append(matrix.shape[0])
        matrix = matrix[1:]
    return tuple(dims)
