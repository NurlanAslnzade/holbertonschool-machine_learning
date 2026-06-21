#!/usr/bin/env python3
"""
Deep RNN forward propagation
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    - rnn_cells: list of RNNCell instances of length l (number of layers)
    - X: numpy.ndarray of shape (t, m, i) containing the input data
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    - h_0: numpy.ndarray of shape (l, m, h) containing initial hidden states
        h: dimensionality of the hidden state

    Returns:
    - H: numpy.ndarray containing all of the hidden states,
         shape (t + 1, l, m, h)
    - Y: numpy.ndarray containing all of the outputs,
         shape (t, m, o) where o is the output dimensionality of the last cell
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize the hidden states array with zeros
    H = np.zeros((t + 1, l, m, h))
    # Place the initial hidden states at time step 0
    H[0] = h_0

    # We don't know the exact output dimension 'o' beforehand,
    # so we figure it out dynamically during the first step or use a list
    Y = []

    # Iterate through each time step
    for step in range(t):
        # The input to the first layer is the current time step's data from X
        current_input = X[step]

        # Iterate through each layer
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # Forward step through the current cell
            h_next, y_next = cell.forward(current_input, h_prev)

            # Store the updated hidden state for this layer at the next step
            H[step + 1, layer] = h_next

            # The current layer's hidden state output becomes the next layer's input
            current_input = h_next

        # Collect the output from the final layer
        Y.append(y_next)

    # Convert output list to a single numpy array
    Y = np.array(Y)

    return H, Y
