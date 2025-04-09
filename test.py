import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf


# Define VAE model with custom train step
    class VAE(keras.Model):

        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return[
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        # custom train step
        def train_step(self, data):

            print("Data Shape:", data.shape)

            with tf.GradientTape() as tape:

                # get the latent representation (run image through the encoder)
                z_mean, z_log_var, z = self.encoder(data)

                print("Z Mean Shape:", z_mean.shape)
                print("Z Shape:", z.shape)

                # form the reconstruction (run latent representation through decoder)
                reconstruction = self.decoder(z)

                print("Reconstruction Shape:", reconstruction.shape)

                # calculate the binary cross entropy reconstruction loss (sum over each pixel and average (mean) across each channel and across the batch)
                reconstruction_loss = ops.mean(
                    ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2),
                    )
                )
                # reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction))
                # reconstruction_loss = ops.mean(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
                # reconstruction_loss = ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
                # reconstruction_loss = reconstruction_loss / (256 * 256)
                # reconstruction_loss = ops.mean(reconstruction_loss)
                # reconstruction_loss = ops.mean(ops.sqrt(keras.losses.mean_squared_error(data, reconstruction)))
                # reconstruction_loss = ops.sqrt(ops.mean(ops.square(data - reconstruction)))

                # reconstruction_loss = data - reconstruction
                # print("Reconstruction Loss Shape:", reconstruction_loss.shape)
                # reconstruction_loss = ops.square(data - reconstruction)
                # print("Reconstruction Loss Shape:", reconstruction_loss.shape)
                # reconstruction_loss = ops.mean(ops.square(data - reconstruction), axis=(1, 2, 3))
                # print("Reconstruction Loss Shape:", reconstruction_loss.shape)
                # reconstruction_loss = ops.sqrt(ops.mean(ops.square(data - reconstruction), axis=(1, 2, 3)))
                # print("Reconstruction Loss Shape:", reconstruction_loss.shape)
                # reconstruction_loss = ops.mean(ops.sqrt(ops.mean(ops.square(data - reconstruction), axis=(1, 2, 3))))
                # print("Reconstruction Loss Shape:", reconstruction_loss.shape)


                # rmse over bce (per pixel)

                # train reconstruction sum over pixel, kl per feature (and with beta)

                # calculate the kl divergence (sum over each latent feature and average (mean) across the batch)
                # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                # kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
                # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                # kl_loss = ops.mean(kl_loss, axis=1)
                # kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                # kl_loss = ops.sum(kl_loss, axis=1) / encoding_dim
                # kl_loss = ops.mean(kl_loss)
                # kl_loss = ops.maximum(kl_loss, 1e-3)
                kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
                kl_loss = ops.mean(kl_loss)


                # 0.01, 0.1

                print("KL Loss Shape:", kl_loss.shape)


                # total loss is the sum of reconstruction loss and kl divergence
                total_loss = reconstruction_loss + kl_loss
                # total_loss = reconstruction_loss + (10 * kl_loss)

                print("Total Loss Shape:", total_loss.shape)


            # gradient decent based on total loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # update loss trackers
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            # return total loss, reconstruction loss and kl divergence
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }






    # define sampling layer
    class Sampling(Layer):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):

            # get the latent distributions
            z_mean, z_log_var = inputs

            # find the batch size and number of latent features (dim)
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]

            # generate the random variables
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)

            # perform reparameterization trick
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon






