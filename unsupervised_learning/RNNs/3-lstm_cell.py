#!/usr/bin/env python3
"""
Module containing the LSTMCell class for forward propagation on a single unit.
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
        # Concatenated feature size for gates: hidden state + input data
        concat_size = h + i

        # Weights initialized with random normal distribution
        self.Wf = np.random.randn(concat_size, h)
        self.Wu = np.random.randn(concat_size, h)
        self.Wc = np.random.randn(concat_size, h)
        self.Wo = np.random.randn(concat_size, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized as zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h) containing previous hidden state
        - c_prev: numpy.ndarray of shape (m, h) containing previous cell state
        - x_t: numpy.ndarray of shape (m, i) containing data input for the cell
          where m is the batch size

        Returns: h_next, c_next, y
        """
        # Concatenate the previous hidden state and current input side-by-side
        # Shape results in (m, h + i)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Gate computations using standard activations
        # Sigmoid function helper: 1 / (1 + exp(-x))
        f_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wf) + self.bf)))
        i_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wu) + self.bu)))

        # Candidate state using tanh activation
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # Next cell state update
        c_next = f_t * c_prev + i_t * c_tilde

        # Output gate and next hidden state computation
        o_t = 1 / (1 + np.exp(-(np.dot(concat, self.Wo) + self.bo)))
        h_next = o_t * np.tanh(c_next)

        # Output projection using Softmax activation
        y_linear = np.dot(h_next, self.Wy) + self.by
        # Softmax step over rows for batched operations
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
