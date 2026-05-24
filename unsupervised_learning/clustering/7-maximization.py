#!/usr/bin/env python3
"""Maximization step in EM algorithm"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm
    for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster

    Returns:
        pi, m, S

        pi: numpy.ndarray of shape (k,) containing the updated priors
        m: numpy.ndarray of shape (k, d) containing the updated means
        S: numpy.ndarray of shape (k, d, d) containing updated
           covariance matrices

        Returns None, None, None on failure
    """

    # Validate X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    # Validate g
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # Check that probabilities sum to 1
    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    # Effective number of points per cluster
    Nk = np.sum(g, axis=1)

    # Updated priors
    pi = Nk / n

    # Updated means
    m = np.dot(g, X) / Nk[:, np.newaxis]

    # Updated covariance matrices
    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = np.dot(weighted_diff.T, diff) / Nk[i]

    return pi, m, S
