#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty (WGAN-GP).

Implements a GAN trained using Wasserstein loss and
gradient penalty for stable learning.
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    WGAN-GP model.
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
        """Initialize model"""
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp

        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, dtype='int32')

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size

        i = 1
        while i < len(self.scal_shape):
            self.scal_shape[i] = 1
            i += 1

        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # --------------------
        # Generator loss
        # --------------------
        def g_loss(x):
            return -tf.reduce_mean(
                self.discriminator(x, training=True)
            )

        self.generator.loss = g_loss

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # --------------------
        # Discriminator loss
        # --------------------
        def d_loss(x, y):
            return (
                tf.reduce_mean(self.discriminator(x, training=True))
                - tf.reduce_mean(self.discriminator(y, training=True))
            )

        self.discriminator.loss = d_loss

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Fake samples"""
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size),
            training=training
        )

    def get_real_sample(self, size=None):
        """Real samples"""
        if not size:
            size = self.batch_size

        idx = tf.random.shuffle(
            tf.range(tf.shape(self.real_examples)[0])
        )[:size]

        return tf.gather(self.real_examples, idx)

    def get_interpolated_sample(self, real, fake):
        """Interpolation"""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real + v * fake

    def gradient_penalty(self, interpolated):
        """Gradient penalty"""
        with tf.GradientTape() as t:
            t.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = t.gradient(pred, interpolated)

        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(grads),
                axis=self.axis
            )
        )

        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, _):
        """Training step"""

        for _ in range(self.disc_iter):

            with tf.GradientTape() as tape:

                real = self.get_real_sample()
                fake = self.get_fake_sample(training=True)

                inter = self.get_interpolated_sample(real, fake)

                d_loss = self.discriminator.loss(fake, real)
                gp = self.gradient_penalty(inter)

                total = d_loss + self.lambda_gp * gp

            grads = tape.gradient(
                total,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:

            fake = self.get_fake_sample(training=True)
            g_loss = self.generator.loss(fake)

        grads = tape.gradient(
            g_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {
            "discr_loss": d_loss,
            "gen_loss": g_loss,
            "gp": gp
        }
