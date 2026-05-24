#!/usr/bin/env python3
"""Bayesian Information Criterion"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: minimum number of clusters
        kmax: maximum number of clusters
        iterations: max iterations for EM
        tol: tolerance for EM
        verbose: boolean for EM verbosity

    Returns:
        best_k, best_result, l, b

        best_k: best value for k based on BIC
        best_result: tuple of (pi, m, S)
        l: log likelihoods for each k
        b: BIC values for each k

        Returns None, None, None, None on failure
    """

    # Validate X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    # Validate kmin
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None

    # Set kmax
    if kmax is None:
        kmax = n

    # Validate kmax
    if (not isinstance(kmax, int) or
            kmax <= 0 or
            kmax < kmin or
            kmax >= n):
        return None, None, None, None

    # Validate iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    # Validate tol
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    # Validate verbose
    if not isinstance(verbose, bool):
        return None, None, None, None

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    results = []

    # Test each k
    for k in range(kmin, kmax + 1):

        pi, m, S, g, lkhd = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        results.append((pi, m, S))
        l[k - kmin] = lkhd

        # Number of parameters
        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)

        # Compute BIC
        b[k - kmin] = p * np.log(n) - (2 * lkhd)

    # Best k is minimum BIC
    best_index = np.argmin(b)

    best_k = best_index + kmin
    best_result = results[best_index]

    return best_k, best_result, l, b
