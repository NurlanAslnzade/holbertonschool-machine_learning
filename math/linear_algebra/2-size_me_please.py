#!/usr/bin/env python3
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
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
