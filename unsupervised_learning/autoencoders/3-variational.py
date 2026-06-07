#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
                       hidden layer in the encoder
        latent_dims: integer containing the dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    # ==================
    # Encoder
    # ==================
    encoder_inputs = keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick"""
        mean, log_var = args

        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]

        epsilon = K.random_normal(shape=(batch, dim))

        return mean + K.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, z_mean, z_log_var]
    )

    # ==================
    # Decoder
    # ==================
    decoder_inputs = keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # ==================
    # Variational Autoencoder
    # ==================
    z, z_mean, z_log_var = encoder(encoder_inputs)

    outputs = decoder(z)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        outputs
    )

    reconstruction_loss *= input_dims

    kl_loss = 1 + z_log_var
    kl_loss -= K.square(z_mean)
    kl_loss -= K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
