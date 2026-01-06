#!/usr/bin/env python3
"""creating dataframe"""
import pandas as pd


def from_numpy(array):
    """creating dataFrame from array"""
    columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:array.shape[1]]
    return pd.DataFrame(array, columns=columns)
