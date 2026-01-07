#!/usr/bin/env python3
"""tasd  dksadmas kms"""


def fill(df):
    """fill dataframe df"""
    df.drop(columns=["Weighted_Price"], inplace=True)
    df["Close"] = df["Close"].fillna(method="ffill")
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
