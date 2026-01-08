#!/usr/bin/env python3
"""ssds sadasdasdas"""
import pandas as pd
index = __import__('10-index').index


def analyze(df):
    """analyzes df"""
    df.drop(columns=["Timestamp"])
    return df.describe()
