#!/usr/bin/env python3
'''
saving and loading config
'''

import tensorflow.keras as K


def save_config(network, filename):
    '''
    network - the model whose configuration should be saved
    filename - the path of the file that the configuration should be saved to
    '''
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    '''
    filename - the file that the configuration should be loaded from
    Returns: the loaded model
    '''
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
