#!/usr/bin/env python3
"""sadasdas dasda"""


def add_matrices2D(mat1, mat2):
    """arraylar cemi"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    new_list = []
    for i in range(len(mat1)):
        new = []
        for k in range(len(mat1[0])):
            new.append(mat1[i][k] + mat2[i][k])
        new_list.append(new)
    return new_list
