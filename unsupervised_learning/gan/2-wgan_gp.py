#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty.
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein Generative Adversarial Network with
    Gradient Penalty (WGAN-GP).
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005,
        lambda_gp=10
    ):
        """
        Initialize the WGAN-GP model.

        Args:
            generator (keras.Model): generator model.
            discriminator (keras.Model): discriminator model.
            latent_generator (callable): latent vector generator.
            real_examples (tf.Tensor): real training samples.
            batch_size (int): batch size.
            disc_iter (int): discriminator iterations per step.
            learning_rate (float): optimizer learning rate.
            lambda_gp (float): gradient penalty coefficient.
        """
        super().__init__()

        # your existing code ...

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples.

        Args:
            size (int, optional): batch size.
            training (bool): training mode.

        Returns:
            tf.Tensor: generated samples.
        """
        # your existing code

    def get_real_sample(self, size=None):
        """
        Return a random batch of real samples.

        Args:
            size (int, optional): batch size.

        Returns:
            tf.Tensor: real samples.
        """
        # your existing code

    def get_interpolated_sample(
        self,
        real_sample,
        fake_sample
    ):
        """
        Generate interpolated samples between
        real and fake samples.

        Args:
            real_sample (tf.Tensor): real batch.
            fake_sample (tf.Tensor): fake batch.

        Returns:
            tf.Tensor: interpolated samples.
        """
        # your existing code

    def gradient_penalty(
        self,
        interpolated_sample
    ):
        """
        Compute the gradient penalty.

        Args:
            interpolated_sample (tf.Tensor):
                interpolated samples.

        Returns:
            tf.Tensor: gradient penalty value.
        """
        # your existing code

    def train_step(self, useless_argument):
        """
        Perform one training step.

        Args:
            useless_argument: unused argument required by Keras.

        Returns:
            dict: training metrics containing:
                - discr_loss
                - gen_loss
                - gp
        """
        # your existing code
