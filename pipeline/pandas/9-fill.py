#!/usr/bin/env python3
"""tasd  dksadmas kms"""


def fill(df):
    """fill dataframe df"""
    df.drop(columns=["Weighted_Price"])
    df = df["Close"].fillna(method="ffill")
    df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(["Close"])
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
