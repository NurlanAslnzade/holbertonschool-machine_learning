#!/usr/bin/env python3
"""salam"""
import pandas as pd


sutun = {'First': [0.0, 0.5, 1.0, 1.5],
       'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame(sutun, index=list("ABCD"))
