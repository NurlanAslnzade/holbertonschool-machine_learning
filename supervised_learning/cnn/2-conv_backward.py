#!/usr/bin/env python3
"""
Convolution forward propagation function
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
    elif padding == "valid":
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_pad = np.pad(
        A_prev,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0)
        ),
        mode="constant"
    )

    h_out = (h_prev + pad_top + pad_bottom - kh) // sh + 1
    w_out = (w_prev + pad_left + pad_right - kw) // sw + 1
    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                hs = h * sh
                ws = w * sw
                a_slice = A_pad[i, hs:hs + kh, ws:ws + kw, :]
                for c in range(c_new):
                    Z[i, h, w, c] = (
                        np.sum(a_slice * W[:, :, :, c]) +
                        float(b[0, 0, 0, c])
                    )

    return activation(Z)
