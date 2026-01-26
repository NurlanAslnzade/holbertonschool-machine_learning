#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices.
    Returns a new matrix, or None if the operation is impossible.
    """
    # Check if inner dimensions match (Cols of mat1 == Rows of mat2)
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with dimensions: len(mat1) x len(mat2[0])
    new_mat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Calculate the dot product of row i and column j
            dot_product = 0
            for k in range(len(mat2)):
                dot_product += mat1[i][k] * mat2[k][j]
            row.append(dot_product)
        new_mat.append(row)

    return new_mat
