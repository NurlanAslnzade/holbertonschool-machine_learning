#!/usr/bin/env python3
"""RNN Cell module."""

import numpy as np


class RNNCell:
    """Represents a simple RNN cell."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Input data of shape (m, i).

        Returns:
            tuple: h_next, y
                h_next is the next hidden state.
                y is the output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
