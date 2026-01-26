#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.
    Returns a new matrix, or None if the operation is impossible.
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new_mat1 = [i[:] for i in mat1]
        new_mat2 = [i[:] for i in mat2]
        return new_mat1 + new_mat2

    if axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
