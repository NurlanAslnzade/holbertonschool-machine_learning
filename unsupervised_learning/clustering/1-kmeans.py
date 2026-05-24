#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Args:
        X: numpy.ndarray of shape (n, d)
        k: number of clusters
        iterations: maximum number of iterations

    Returns:
        C, clss
        C: centroid means
        clss: index of cluster for each data point
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2):
        return None, None

    if (not isinstance(k, int) or k <= 0):
        return None, None

    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # initialize centroids
    C = np.random.uniform(low, high, (k, d))

    for i in range(iterations):

        old_C = np.copy(C)

        # compute distances
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)

        # assign clusters
        clss = np.argmin(distances, axis=0)

        # update centroids
        for j in range(k):

            points = X[clss == j]

            if len(points) == 0:
                C[j] = np.random.uniform(low, high, (1, d))
            else:
                C[j] = np.mean(points, axis=0)

        # stop if converged
        if np.allclose(old_C, C):
            break

    return C, clss
