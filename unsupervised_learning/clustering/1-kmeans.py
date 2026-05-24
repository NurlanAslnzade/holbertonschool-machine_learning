#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number
                    of iterations that should be performed

    Returns:
        C, clss

        C: numpy.ndarray of shape (k, d) containing the centroid
           means for each cluster

        clss: numpy.ndarray of shape (n,) containing the index
               of the cluster in C that each data point belongs to

        Returns None, None on failure
    """

    # Validate inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    # Initialize centroids
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    C = np.random.uniform(X_min, X_max, (k, d))

    # Run K-means
    for i in range(iterations):

        old_C = np.copy(C)

        # Compute distances from points to centroids
        distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))

        # Assign each point to closest centroid
        clss = np.argmin(distances, axis=0)

        # Update centroids
        for j in range(k):

            if X[clss == j].size == 0:
                C[j] = np.random.uniform(X_min, X_max, (1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        # Stop if centroids do not change
        if np.allclose(old_C, C):
            break

    return C, clss
