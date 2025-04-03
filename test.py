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


encoding_dim = 24
run = 2
epochs = 750
batch_size = 32

# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")


# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

# load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_" + str(run) + ".npy")[0]
extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
encoding_dim = extracted_features.shape[1]
extracted_features_switch = extracted_features.T

# print(extracted_features.shape)

extracted_features = extracted_features[:len(all_properties)]


print(extracted_features.shape)
disk_structures = np.array(all_properties["n_r"] <= 2.5)
extracted_features = extracted_features[disk_structures]
print(extracted_features.shape)









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

        with tf.GradientTape() as tape:

            # get the latent representation (run image through the encoder)
            z_mean, z_log_var, z = self.encoder(data)

            # form the reconstruction (run latent representation through decoder)
            reconstruction = self.decoder(z)

            # calculate the binary cross entropy reconstruction loss (sum over each pixel and average (mean) across each channel and across the batch)
            reconstruction_loss = ops.sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2))
            reconstruction_loss = reconstruction_loss / (256 * 256)
            reconstruction_loss = ops.mean(reconstruction_loss)

            # calculate the kl divergence (sum over each latent feature and average (mean) across the batch)
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.sum(kl_loss, axis=1) / encoding_dim
            kl_loss = ops.mean(kl_loss)

            # total loss is the sum of reconstruction loss and kl divergence
            total_loss = reconstruction_loss + (10 * kl_loss)


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


