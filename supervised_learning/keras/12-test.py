#!/usr/bin/env python3
'''
testing the model
'''

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''
    network - the model to test
    data - the input data to test the model with
    labels - the one-hot labels of data
    verbose - is a boolean that determines if output should be printed during
              the testing process
    Returns: the loss and accuracy of the model with the testing data
    '''

    return network.evaluate(data, labels, verbose=verbose)
