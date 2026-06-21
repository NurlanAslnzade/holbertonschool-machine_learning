#!/usr/bin/env python3
"""
Module containing the LSTMCell class for forward propagation
on a single unit.
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit.
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Parameters:
        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        """
        concat_size = h + i

        self.Wf = np.random.randn(concat_size, h)
        self.Wu = np.random.randn(concat_size, h)
        self.Wc = np.random.randn(concat_size, h)
        self.Wo = np.random.randn(concat_size, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: ndarray of shape (m, h)
          containing previous hidden state
        - c_prev: ndarray of shape (m, h)
          containing previous cell state
        - x_t: ndarray of shape (m, i)
          containing input data
        - m: batch size

        Returns:
        - h_next: next hidden state
        - c_next: next cell state
        - y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_t = 1 / (
            1 + np.exp(-(np.dot(concat, self.Wf) + self.bf))
        )

        i_t = 1 / (
            1 + np.exp(-(np.dot(concat, self.Wu) + self.bu))
        )

        c_tilde = np.tanh(
            np.dot(concat, self.Wc) + self.bc
        )

        c_next = f_t * c_prev + i_t * c_tilde

        o_t = 1 / (
            1 + np.exp(-(np.dot(concat, self.Wo) + self.bo))
        )

        h_next = o_t * np.tanh(c_next)

        y_linear = np.dot(h_next, self.Wy) + self.by

        exp_y = np.exp(
            y_linear - np.max(y_linear, axis=1, keepdims=True)
        )

        y = exp_y / np.sum(
            exp_y, axis=1, keepdims=True
        )

        return h_next, c_next, y
