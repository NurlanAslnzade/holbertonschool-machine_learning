#!/usr/bin/env python3
"""sadasdas dasda"""


def add_arrays(arr1, arr2):
    """arraylar cemi"""
    if len(arr1) != len(arr2):
        return None
    new_list = []
    for i in range(len(arr1)):
        new_list.append(arr1[i] + arr2[i])
    return new_list
