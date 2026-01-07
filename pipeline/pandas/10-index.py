#!/usr/bin/env python3
"""setting new index"""


def index(df):
    df = df.set_index("Timestamp")
