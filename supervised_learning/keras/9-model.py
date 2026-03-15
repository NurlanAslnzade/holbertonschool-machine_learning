#!/usr/bin/env python3
'''
model
'''


import tensorflow.keras as K


def save_model(network, filename):
    '''
    network - the model to save
    filename - the path of the file that the model should be saved to

    returns: None
    '''

    network.save(filename)


def load_model(filename):
    '''
    filename - the path of the file that the model should be loaded from

    returns: the loaded model
    '''

    return K.models.load_model(filename)
