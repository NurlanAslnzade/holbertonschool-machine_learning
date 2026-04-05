#!/usr/bin/env python3
"""
Pooling forward propagation function
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the kernel size.
        stride: tuple of (sh, sw) containing the strides.
        mode: string containing either 'max' or 'avg' pooling.

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                hs = h * sh
                ws = w * sw
                a_slice = A_prev[i, hs:hs + kh, ws:ws + kw, :]
                if mode == 'max':
                    A[i, h, w, :] = np.max(a_slice, axis=(0, 1))
                elif mode == 'avg':
                    A[i, h, w, :] = np.mean(a_slice, axis=(0, 1))

    return A
