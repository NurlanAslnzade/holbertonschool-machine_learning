#!/usr/bin/env/python3
def matrix_shape(matrix):
    a = []
    if isinstance(matrix[0], list):
        a.append(len(matrix))
        a.append(len(matrix[0]))
        a.append(len(matrix[0][0]))
    else:
        a.append(len(matrix))
        a.append(len(matrix[0]))
    return a
print(matrix_shape(mat2))
