#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix.
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): matrix to invert.

    Returns:
        list of lists of floats: inverse of matrix,
        or None if matrix is singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if (not isinstance(matrix, list) or
            matrix == [] or
            not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = _determinant(matrix)
    if det == 0:
        return None

    # 1x1 xüsusi hal
    if n == 1:
        return [[1 / det]]

    adj = _adjoint(matrix)
    return [[adj[i][j] / det for j in range(n)] for i in range(n)]


def _determinant(matrix):
    """
    Helper function that calculates determinant of a square matrix.
    """
    n = len(matrix)

    # 1x1
    if n == 1:
        return matrix[0][0]

    # 2x2
    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    # nxn: Laplace expansion on first row
    det = 0
    for j in range(n):
        det += ((-1) ** j) * matrix[0][j] * _determinant(
            _minor(matrix, 0, j)
        )
    return det


def _minor(matrix, row, col):
    """
    Build the minor matrix removing one row and one column.
    """
    return [r[:col] + r[col + 1:]
            for i, r in enumerate(matrix) if i != row]


def _adjoint(matrix):
    """
    Compute the adjugate (adjoint) of a square matrix.
    """
    n = len(matrix)

    if n == 1:
        return [[1]]

    adj = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            m = _minor(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * _determinant(m)
            # transpose while filling
            adj[j][i] = cofactor

    return adj
