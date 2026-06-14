#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.
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
        self.axis = tf.range(1, self.len_dims, dtype="int32")

        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size

        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1

        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Generator loss
        self.generator.loss = (
            lambda fake:
            -tf.math.reduce_mean(self.discriminator(fake))
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss
        self.discriminator.loss = (
            lambda fake, real:
            tf.math.reduce_mean(self.discriminator(fake))
            - tf.math.reduce_mean(self.discriminator(real))
        )

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
        """
        Generate fake samples.
        """
        if not size:
            size = self.batch_size

        latent = self.latent_generator(size)
        return self.generator(latent, training=training)

    def get_real_sample(self, size=None):
        """
        Draw a random batch of real samples.
        """
        if not size:
            size = self.batch_size

        idx = tf.random.shuffle(
            tf.range(tf.shape(self.real_examples)[0])
        )[:size]

        return tf.gather(self.real_examples, idx)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate samples interpolated between
        real and fake samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u

        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)

            pred = self.discriminator(
                interpolated_sample,
                training=True
            )

        grads = gp_tape.gradient(
            pred,
            [interpolated_sample]
        )[0]

        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(grads),
                axis=self.axis
            )
        )

        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, _):
        """
        One training step.
        """

        for _ in range(self.disc_iter):

            with tf.GradientTape() as tape:

                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                interpolated = self.get_interpolated_sample(
                    real_sample,
                    fake_sample
                )

                discr_loss = self.discriminator.loss(
                    fake_sample,
                    real_sample
                )

                gp = self.gradient_penalty(interpolated)

                new_discr_loss = (
                    discr_loss + self.lambda_gp * gp
                )

            grads = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(
                    grads,
                    self.discriminator.trainable_variables
                )
            )

        with tf.GradientTape() as tape:

            fake_sample = self.get_fake_sample(training=True)

            gen_loss = self.generator.loss(fake_sample)

        grads = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(
                grads,
                self.generator.trainable_variables
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp
        }

    def replace_weights(self, gen_h5, disc_h5):
        """
        Load pretrained generator and discriminator
        weights from HDF5 files.

        Args:
            gen_h5 (str): generator weights file
            disc_h5 (str): discriminator weights file
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
