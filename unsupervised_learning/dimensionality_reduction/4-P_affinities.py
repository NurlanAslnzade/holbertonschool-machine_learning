#!/usr/bin/env python3
"""Calculates symmetric P affinities"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities
    of a data set
    """

    n, d = X.shape

    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):

        # Remove self-distance
        Di = np.delete(D[i], i)

        # Initial entropy and probabilities
        Hi, Pi = HP(Di, betas[i, 0])

        Hdiff = Hi - H

        beta_min = None
        beta_max = None

        # Binary search
        while np.abs(Hdiff) > tol:

            if Hdiff > 0:
                beta_min = betas[i, 0]

                if beta_max is None:
                    betas[i, 0] *= 2.
                else:
                    betas[i, 0] = (
                        betas[i, 0] + beta_max
                    ) / 2.

            else:
                beta_max = betas[i, 0]

                if beta_min is None:
                    betas[i, 0] /= 2.
                else:
                    betas[i, 0] = (
                        betas[i, 0] + beta_min
                    ) / 2.

            Hi, Pi = HP(Di, betas[i, 0])
            Hdiff = Hi - H

        # Insert 0 probability for self
        P[i] = np.insert(Pi, i, 0)

    # Symmetrize
    P = (P + P.T) / (2 * n)

    return P
