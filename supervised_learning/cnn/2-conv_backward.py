#!/usr/bin/env python3
"""
Convolution backward propagation function
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
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

    A_prev_pad = np.pad(
        A_prev,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0)
        ),
        mode="constant"
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros((1, 1, 1, c_new))

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                hs = h * sh
                ws = w * sw
                for c in range(c_new):
                    dz = dZ[i, h, w, c]
                    a_slice = a_prev_pad[hs:hs + kh, ws:ws + kw, :]
                    da_prev_pad[hs:hs + kh, ws:ws + kw, :] += (
                        W[:, :, :, c] * dz
                    )
                    dW[:, :, :, c] += a_slice * dz
                    db[0, 0, 0, c] += dz
        dA_prev_pad[i] = da_prev_pad

    if padding == "same":
        dA_prev = dA_prev_pad[
            :,
            pad_top:h_prev + pad_top,
            pad_left:w_prev + pad_left,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
