#!/usr/bin/env python3
"""
Module for horizontal image flipping.

Contains a function to flip a 3D TensorFlow image tensor horizontally.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    The image is flipped along the width dimension (left ↔ right).

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            containing the image to flip.

    Returns:
        tf.Tensor: A 3D tensor of the same shape as `image`, flipped horizontally.
    """
    return tf.image.flip_left_right(image)
