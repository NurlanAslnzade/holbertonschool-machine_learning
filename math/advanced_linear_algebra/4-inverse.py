#!/usr/bin/env python3
"""
Module to calculate the determinant and inverse of a matrix.
"""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    Args:
        matrix (list[list[int|float]]): Square matrix.

    Returns:
        int | float: Determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    _validate_matrix(matrix)

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for col in range(n):
        minor = _minor(matrix, 0, col)
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)
    return det


def inverse(matrix):
    """
    Calculate the inverse of a matrix.

    Args:
        matrix (list[list[int|float]]): Square matrix.

    Returns:
        list[list[float]] | None: Inverse of matrix,
        or None if matrix is singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    _validate_matrix(matrix)

    det = determinant(matrix)
    if det == 0:
        return None

    n = len(matrix)
    adj = _adjoint(matrix)

    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i]
