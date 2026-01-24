#!/usr/bin/env python3
"""asdasd asda"""


def matrix_transpose(matrix):
    """asdasdas"""
    rows = len(matrix)
    cols = len(matrix[0])
    new = []
    for i in range(cols):
        row = []
        for k in range(rows):
            row.append(matrix[k][i])
        new.append(row)
    return new
