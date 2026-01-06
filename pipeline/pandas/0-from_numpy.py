#!/usr/bin/env python3
"""salam"""
import pandas as pd


def from_numpy(array):
    """salam2"""
    columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:array.shape[1]]
    return pd.DataFrame(array, columns=columns)
