#!/usr/bin/env python3
"""
3-generate_faces.py

Defines a convolutional Generator and Discriminator suitable for
training a GAN on 16x16 grayscale face images.

The generator:
    - Input shape: (16,)
    - Output shape: (16, 16, 1)

The discriminator:
    - Input shape: (16, 16, 1)
    - Output shape: (1,)

All activations use tanh as required.
All Conv2D layers use padding="same".
"""

from tensorflow import keras


def convolutional_GenDiscr():
    """
    Build and return the generator and discriminator models.

    Returns:
        tuple:
            generator (keras.Model)
            discriminator (keras.Model)
    """

    def get_generator():
        """
        Create the generator network.

        Architecture:

        Input (16,)
            ↓
        Dense(2048)
            ↓
        Reshape((2, 2, 512))
            ↓
        UpSampling2D
            ↓
        Conv2D(64)
            ↓
        BatchNormalization
            ↓
        tanh
            ↓
        UpSampling2D
            ↓
        Conv2D(16)
            ↓
        BatchNormalization
            ↓
        tanh
            ↓
        UpSampling2D
            ↓
        Conv2D(1)
            ↓
        BatchNormalization
            ↓
        tanh
        """

        inputs = keras.Input(shape=(16,))

        x = keras.layers.Dense(2048, activation="tanh")(inputs)

        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            16,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(
            1,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)

        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(
            inputs,
            outputs,
            name="generator"
        )

    def get_discriminator():
        """
        Create the discriminator network.

        Architecture:

        Input (16,16,1)
            ↓
        Conv2D(32)
            ↓
        MaxPool
            ↓
        tanh
            ↓
        Conv2D(64)
            ↓
        MaxPool
            ↓
        tanh
            ↓
        Conv2D(128)
            ↓
        MaxPool
            ↓
        tanh
            ↓
        Conv2D(256)
            ↓
        MaxPool
            ↓
        tanh
            ↓
        Flatten
            ↓
        Dense(1, tanh)
        """

        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same"
        )(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same"
        )(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(
            1,
            activation="tanh"
        )(x)

        return keras.Model(
            inputs,
            outputs,
            name="discriminator"
        )

    return get_generator(), get_discriminator()
