#!/usr/bin/env python3
"""ssds"""
import pandas as pd


def from_file(filename, delimiter):
    """salam"""
    return pd.read_csv(filename, sep=delimiter)
