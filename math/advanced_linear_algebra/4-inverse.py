#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""

def inverse(matrix):
    """
    Calculates the inverse of a matrix.
    """
    if not isinstance(matrix, list) or not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    det = _determinant(matrix)
    if det == 0:
        return None
    
    adj = _adjoint(matrix)
    inv = [[round((adj[j][i] / det), 16) if det != 0 else 0 for j in range(n)] for i in range(n)]
    return inv

def _determinant(mat):
    n = len(mat)
    if n == 1:
        return mat[0][0]
    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    det = 0
    for c in range(n):
        minor = _minor(mat, 0, c)
        det += ((-1) ** c) * mat[0][c] * _determinant(minor)
    return det

def _minor(mat, i, j):
    return [row[:j] + row[j+1:] for row in (mat[:i] + mat[i+1:])]

def _adjoint(mat):
    n = len(mat)
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = _minor(mat, i, j)
            adj[j][i] = ((-1) ** (i + j)) * _determinant(minor)
    return adj
