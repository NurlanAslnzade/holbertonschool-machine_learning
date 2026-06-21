#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: previous hidden state (m, h)
        x_t: input data (m, i)

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))

        # Reset gate
        r = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))

        # Candidate hidden state
        concat_h = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(concat_h @ self.Wh + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_hat

        # Output
        y_linear = h_next @ self.Wy + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
