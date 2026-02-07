#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): square matrix of shape (n, n).

    Returns:
        str or None: one of
            'Positive definite',
            'Positive semi-definite',
            'Negative semi-definite',
            'Negative definite',
            'Indefinite',
            or None if matrix is invalid or does not fit.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Empty or not 2D or not square → invalid
    if matrix.size == 0 or matrix.ndim != 2:
        return None
    n, m = matrix.shape
    if n != m:
        return None

    # Öz dəyərləri hesablayaq
    try:
        eigvals = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # Kiçik sayısal xətaları sıfıra yaxınlaşdırmaq üçün
    tol = 1e-8
    eig_real = eigvals.real  # matris realdır, xəyali hissə ~0 olur
    eig_real[np.abs(eig_real) < tol] = 0.0

    pos = np.any(eig_real > 0)
    neg = np.any(eig_real < 0)
    all_pos = np.all(eig_real > 0)
    all_neg = np.all(eig_real < 0)
    all_nonneg = np.all(eig_real >= 0)
    all_nonpos = np.all(eig_real <= 0)

    if all_pos:
        return "Positive definite"
    if all_neg:
        return "Negative definite"
    if all_nonneg and pos:
        return "Positive semi-definite"
    if all_nonpos and neg:
        return "Negative semi-definite"
    if pos and neg:
        return "Indefinite"

    return None
