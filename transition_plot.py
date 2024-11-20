import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.config.list_physical_devices('GPU')


# number of extracted features
encoding_dim = 30

# # select which gpu to use
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

# number of epochs for run
epochs = 300


# # normalise each band individually
# def normalise_independently(image):
#     image = image.T
#     for i in range(0, 3):
#         image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
#     return image.T
#
# # normalise each band to r
# def normalise_to_r(image):
#     image = image.T
#     for i in range(0, 3):
#         image[i] = (image[i] - np.min(image[i])) / (np.max(image[1]) - np.min(image[1]))
#     return image.T
#
#
#
# # list to contain all galaxy images
# all_images = []
#
# # load the ids of the chosen galaxies
# chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/Chosen Galaxies.npy")
#
#
#
# # # loop through each galaxy in the supplemental file
# for i, galaxy in enumerate(chosen_galaxies):
#
#     # get the filename of each galaxy in the supplemental file
#     filename = "galrand_" + str(galaxy) + ".png"
#
#     # open the image and append it to the main list
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
#
#     # find smallest non zero pixel value in the image and replace all zero values with this (for log transformation)
#     smallest_non_zero = np.min(image[image > 0])
#     image = np.where(image == 0.0, smallest_non_zero, image)
#
#     # apply log transformation to the image
#     # image = np.log10(image)
#
#     # normalise the image (either each band independently or to the r band)
#     image = normalise_independently(image)
#     # image = normalise_to_r(image)
#
#     # add the image to the dataset
#     all_images.append(image)
#
#
#
#
#
#
# # split the data into training and testing data (200 images used for testing)
# train_images = np.array(all_images[:-200])
# test_images = np.array(all_images[-200:])






# Define VAE model with custom time step
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss + kl_loss))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

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
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon





# encoding_dim = 20




# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (8, 8, 4)
x = Flatten()(x)                                                                                                # (256)
x = Dense(units=64)(x)                                                                                          # (64)
z_mean = Dense(encoding_dim, name="z_mean")(x)
z_log_var = Dense(encoding_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Define keras tensor for the decoder
latent_input = keras.Input(shape=(encoding_dim,))

# layers for the decoder
x = Dense(units=64)(latent_input)                                                                               # (64)
x = Dense(units=256)(x)                                                                                         # (256)
x = Reshape((8, 8, 4))(x)                                                                                       # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)    # (128, 128, 3) (forces output between 0 and 1)
# decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same", name="decoded")(x)     # (128, 128, 3) (forces output to be positive)

# build the decoder
decoder = keras.Model(latent_input, decoded, name="decoder")
decoder.summary()




# build and compile the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())





# or load the weights for the model
vae.load_weights("Variational Eagle/Weights/Normalised to g/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_1.weights.h5")

# load the extracted features
extracted_features = np.load("Variational Eagle/Extracted Features/Normalised to g/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_2.npy")[0]

# list of medians for all extracted features
med_extracted_features = [np.median(extracted_features.T[i]) for i in range(encoding_dim)]


# apply pca on the extracted features and project the extracted features
pca = PCA(n_components=11).fit(extracted_features)
pca_features = pca.transform(extracted_features)

print(pca_features.shape)

# list of medians for all pca features
med_pca_features = [np.median(pca_features.T[i]) for i in range(11)]

print(med_pca_features)



