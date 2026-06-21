#!/usr/bin/env python3
"""RNN forward propagation module."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell.
        X (numpy.ndarray): Data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (m, h).

    Returns:
        tuple: H, Y
            H contains all hidden states.
            Y contains all outputs.
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
