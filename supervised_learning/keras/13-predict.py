#!/usr/bin/env python3
'''
model prediction
'''

import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''
    network - model to make the prediction with
    data - the input data to make the prediction with
    verbose - a boolean that determines if the results should be printed
    '''

    return network.predict(data, verbose=verbose)
