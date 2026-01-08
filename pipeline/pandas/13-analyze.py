#!/usr/bin/env python3
"""ssds sadasdasdas"""


def analyze(df):
    """analyzes df"""
    df.drop(columns=["Timestamp"])
    return df.describe()
