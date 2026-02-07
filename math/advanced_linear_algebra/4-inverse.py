#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
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


def adjugate(matrix):
    """Calculates the adjugate (transpose of cofactor matrix)."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(
            len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    adjugate_mat = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [r[:j] + r[j + 1:]
                         for r in (matrix[:i] + matrix[i + 1:])]
            minor_val = determinant(submatrix)
            sign = 1 if (i + j) % 2 == 0 else -1
            row.append(sign * minor_val)
        adjugate_mat.append(row[::-1])  # Reverse row for transpose equiv
    return adjugate_mat


def inverse(matrix):
    """Calculates matrix inverse using adjugate/determinant formula."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(
            len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    n = len(matrix)
    inverse_mat = []
    for i in range(n):
        row = [round(adj[i][j] / det, 14) for j in range(n)]
        inverse_mat.append(row)
    return inverse_mat
