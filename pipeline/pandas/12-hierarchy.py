#!/usr/bin/env python3
"""ssds sadasdasdas"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """sadasdasd"""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[1417411980:1417417980]
    df1 = df1.loc[1417411980:1417417980]
    df = pd.concat([df1, df2], keys = ["bitstamp", "coinbase"])
    return df.loc[::-1].sort_index()
