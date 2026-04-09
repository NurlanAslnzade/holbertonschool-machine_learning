#!/usr/bin/env python3
"""
Module for random crop image augmentation.

Contains a function to randomly crop a 3D TensorFlow image tensor.
"""


import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    The crop dimensions are given by `size`; the position is chosen randomly.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            containing the image to crop.
        size (tuple): A tuple (crop_height, crop_width, channels) specifying
            the size of the crop.

    Returns:
        tf.Tensor: A 3D tensor of shape `size`, containing the randomly
            cropped image.
    """
    return tf.image.random_crop(image, size=size)
