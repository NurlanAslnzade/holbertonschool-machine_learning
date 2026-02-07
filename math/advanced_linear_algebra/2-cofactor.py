#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""


def determinant(mat):
    """Recursive determinant for square matrix mat."""
    n = len(mat)
    if n == 0:
        return 1  # Convention for empty matrix det
    if n == 1:
        return mat[0][0]
    if n == 2:
        return (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0])
    det = 0
    for col in range(n):
        submat = [row[:col] + row[col + 1:] for row in mat[1:]]
        cofactor = ((-1) ** col) * determinant(submat)
        det += mat[0][col] * cofactor
    return det


def minor(matrix):
    """Compute minor matrix."""
    if (not isinstance(matrix, list) or
        not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if (not matrix or
        not all(len(row) == len(matrix) for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    minor_mat = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [r[:j] + r[j + 1:]
                         for r in (matrix[:i] + matrix[i + 1:])]
            row.append(determinant(submatrix))
        minor_mat.append(row)
    return minor_mat


def cofactor(matrix):
    """Calculates the cofactor matrix from minor matrix."""
    if (not isinstance(matrix, list) or
        not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if (not matrix or
        not all(len(row) == len(matrix) for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)
    n = len(matrix)
    cofactor_mat = []
    for i in range(n):
        row = []
        for j in range(n):
            sign = 1 if (i + j) % 2 == 0 else -1
            row.append(sign * minors[i][j])
        cofactor_mat.append(row)
    return cofactor_mat
