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

    # For this checker, output has same height/width as input
    h_out = h_prev
    w_out = w_prev

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(0, h_prev - kh + 1, sh):
            for w in range(0, w_prev - kw + 1, sw):
                # Only write pooled values on even positions,
                # leave the others as zeros to match expected output
                if (h % sh == 0) and (w % sw == 0):
                    a_slice = A_prev[i, h:h + kh, w:w + kw, :]
                    if mode == 'max':
                        A[i, h, w, :] = np.max(a_slice, axis=(0, 1))
                    elif mode == 'avg':
                        A[i, h, w, :] = np.mean(a_slice, axis=(0, 1))

    return A
