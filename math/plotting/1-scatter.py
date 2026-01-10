#!/usr/bin/env python3
"""plot yaratmaq"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """scatter plot yaratmaq"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000)
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, marker='o', color='magenta', linestyle="None")
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
