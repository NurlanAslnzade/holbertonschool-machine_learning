#!/usr/bin/env python3
"""ssds sadasdasdas"""
import pandas as pd


def rename(df):
    """asdasdas dasdasdasdas"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit="s")
    return df.loc[:, ["Datetime", "Close"]]
