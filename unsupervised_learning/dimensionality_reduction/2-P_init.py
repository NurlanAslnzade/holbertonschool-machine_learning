#!/usr/bin/env python3
"""Initializes variables for t-SNE"""


import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables used to calculate
    P affinities in t-SNE

    Parameters:
    X: numpy.ndarray of shape (n, d)
    perplexity: perplexity value

    Returns:
    D, P, betas, H
    """

    n, d = X.shape

    # Compute squared norms
    sum_X = np.sum(np.square(X), axis=1)

    # Compute pairwise squared Euclidean distances
    D = np.add(
        np.add(-2 * np.dot(X, X.T), sum_X).T,
        sum_X
    )

    # Ensure diagonal is exactly 0
    np.fill_diagonal(D, 0)

    # Initialize P matrix
    P = np.zeros((n, n))

    # Initialize betas
    betas = np.ones((n, 1))

    # Shannon entropy
    H = np.log2(perplexity)

    return D, P, betas, H
