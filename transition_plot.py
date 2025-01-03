import os
from matplotlib.pyplot import figure
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
import random




# number of extracted features
encoding_dim = 20

run = 3

# number of epochs for run
epochs = 300




# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T




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

# load the weights
vae.load_weights("Variational Eagle/Weights/Normalised Individually/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_" + str(run) + ".weights.h5")










# for a partially balanced dataset

# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# print(bad_fit)

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# get the indices of the different types of galaxies (according to sersic index)
spirals_indices = list(all_properties.loc[all_properties["n_r"] <= 2.5].index)
unknown_indices = list(all_properties.loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index)
ellipticals_indices = list(all_properties.loc[all_properties["n_r"] >= 4].index)

# randomly sample half the spiral galaxies
random.seed(1)
chosen_spiral_indices = random.sample(spirals_indices, round(len(spirals_indices)/2))

# indices of the galaxies trained on the model that we have properties for
chosen_indices = chosen_spiral_indices + unknown_indices + ellipticals_indices

# reorder the properties dataframe to match the extracted features of the balanced dataset
all_properties = all_properties.loc[chosen_indices]

# get the indices of the randomly sampled testing set (from the full dataset with augmented images)
random.seed(2)
dataset_size = len(chosen_spiral_indices) + len(unknown_indices) + (4 * len(ellipticals_indices))
test_indices = random.sample(range(0, dataset_size), 20)

# flag the training set in the properties dataframe (removing individually effects the position of the other elements)
for i in test_indices:
    if i <= len(all_properties):
        all_properties.iloc[i] = np.nan

# remove the training set from the properties dataframe
all_properties = all_properties.dropna()


# load the extracted features
extracted_features = np.load("Variational Eagle/Extracted Features/Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
extracted_features_switch = extracted_features.T

# perform pca on the extracted features
# pca = PCA(n_components=13).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T

# get the indices of the different types of galaxies (according to sersic index) after restructuring of properties dataframe
spirals_indices = list(all_properties.loc[all_properties["n_r"] <= 2.5].index)
unknown_indices = list(all_properties.loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index)
ellipticals_indices = list(all_properties.loc[all_properties["n_r"] >= 4].index)

# split the extracted features array into the half with spirals and unknown and ellipticals
extracted_features_spiral_unknown = extracted_features[:(len(spirals_indices) + len(unknown_indices))]
extracted_features_elliptical = extracted_features[(len(spirals_indices) + len(unknown_indices)):]

# remove the augmented images (3 of every 4 elliptical galaxies)
extracted_features_elliptical = np.array([extracted_features_elliptical[i] for i in range(len(extracted_features_elliptical)) if i % 4 == 0])

# join the two arrays back together
extracted_features = np.array(list(extracted_features_spiral_unknown) + list(extracted_features_elliptical))
extracted_features_switch = extracted_features.T


print(len(extracted_features.T))


num_varying_features = 15

chosen_feature = 1

med_pca_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]

print(len(med_pca_features))

# varying_feature_values = np.linspace(np.min(extracted_features.T[chosen_feature]), np.max(extracted_features.T[chosen_feature]), num_varying_features)

# fig, axs = plt.subplots(1, num_varying_features, figsize=(num_varying_features, 3))
fig, axs = plt.subplots(len(extracted_features.T), num_varying_features, figsize=(num_varying_features, len(extracted_features)))

for i in range(len(extracted_features.T)):

    varying_feature_values = np.linspace(np.min(extracted_features.T[i]), np.max(extracted_features.T[i]), num_varying_features)

    for j in range(num_varying_features):

        temp_pca_features = med_pca_features.copy()
        temp_pca_features[i] = varying_feature_values[j]

        temp_features = temp_pca_features
        temp_features = np.expand_dims(temp_features, axis=0)

        # temp_features = pca.inverse_transform(temp_pca_features)
        # temp_features = np.expand_dims(temp_features, axis=0)

        reconstruction = vae.decoder.predict(temp_features)[0]

        axs[i][j].imshow(reconstruction)
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)


plt.show()






# # list of medians for all extracted features
# med_extracted_features = [np.median(extracted_features.T[i]) for i in range(encoding_dim)]
#
# # values to vary the chosen extracted feature (equally spaced values between min and max)
# # varying_feature_values = np.linspace(np.min(extracted_features.T[chosen_feature]), np.max(extracted_features.T[chosen_feature]), num_varying_features)
# varying_feature_values = [-5, -3] + list(np.linspace(-2.5, 1.5, 12)) + [2.5]
# # varying_feature_values = [-4, -3] + list(np.linspace(-2, 3, 13))
# # varying_feature_values = [-5, -3] + list(np.linspace(-2.5, 2, 12)) + [2.5]
#
# # second feature
# # varying_feature_values_2 = np.linspace(np.min(extracted_features.T[chosen_feature_2]), np.max(extracted_features.T[chosen_feature_2]), num_varying_features)
# varying_feature_values_2 = [-2.5] + list(np.linspace(-2, 2, 12)) + [2.5, 3.5]
# varying_feature_values_2 = varying_feature_values_2[::-1]
# # varying_feature_values_2 = [-0.3, -0.2] + list(np.linspace(-0.15, 0.05, 12)) + [0.1]
# # varying_feature_values_2 = [-0.1, -0.8] + list(np.linspace(-0.075, 0.1, 12)) + [0.15]
#
#
#
# # apply pca on the extracted features and project the extracted features
# pca = PCA(n_components=13).fit(extracted_features)
# pca_features = pca.transform(extracted_features)
#
# # list of medians for all pca features (equally spaced values between min and max)
# med_pca_features = [np.median(pca_features.T[i]) for i in range(11)]
#
# # values to vary the chosen pca feature
# varying_pca_feature_values = np.linspace(np.min(pca_features.T[chosen_pca_feature]), np.max(pca_features.T[chosen_pca_feature]), num_varying_features)
# # varying_pca_feature_values = [-2.3] + list(np.linspace(-2, 2.5, 14))
# # varying_pca_feature_values = [2] + list(np.linspace(-1.25, 2.5, 13)) + [3]
# # varying_pca_feature_values = varying_pca_feature_values[::-1]
# # varying_pca_feature_values = [-4, -3] + list(np.linspace(-2, 2, 13))
#
#
#
# fig, axs = plt.subplots(5, num_varying_features, figsize=(15, 6))
#
# for i in range(num_varying_features):
#
#     # 8, 10
#     temp_features = med_extracted_features.copy()
#     temp_features[chosen_feature] = varying_feature_values[i]
#     temp_features = np.expand_dims(temp_features, axis=0)
#
#     reconstruction = vae.decoder.predict(temp_features)[0]
#     reconstruction = normalise_independently(reconstruction)
#
#     axs[0, i].imshow(reconstruction)
#     axs[0, i].get_xaxis().set_visible(False)
#     axs[0, i].get_yaxis().set_visible(False)
#
#
#
#     temp_features_2 = med_extracted_features.copy()
#     temp_features_2[chosen_feature_2] = varying_feature_values_2[i]
#     temp_features_2 = np.expand_dims(temp_features_2, axis=0)
#
#     reconstruction_2 = vae.decoder.predict(temp_features_2)[0]
#     reconstruction_2 = normalise_independently(reconstruction_2)
#
#     # axs[1, i].imshow(reconstruction_2)
#     axs[1, i].get_xaxis().set_visible(False)
#     axs[1, i].get_yaxis().set_visible(False)
#
#
#
#     temp_features_3 = med_extracted_features.copy()
#     temp_features_3[chosen_feature] = varying_feature_values[i]
#     temp_features_3[chosen_feature_2] = varying_feature_values_2[i]
#     temp_features_3 = np.expand_dims(temp_features_3, axis=0)
#
#     reconstruction_3 = vae.decoder.predict(temp_features_3)[0]
#     reconstruction_3 = normalise_independently(reconstruction_3)
#
#     # axs[2, i].imshow(reconstruction_3)
#     axs[2, i].get_xaxis().set_visible(False)
#     axs[2, i].get_yaxis().set_visible(False)
#
#
#
#     fig.delaxes(axs[3, i])
#
#
#     # 1
#     temp_pca_features = med_pca_features.copy()
#     temp_pca_features[chosen_pca_feature] = varying_pca_feature_values[-1 * i]
#     temp_pca_features = pca.inverse_transform(temp_pca_features)
#     temp_pca_features = np.expand_dims(temp_pca_features, axis=0)
#
#     reconstruction = vae.decoder.predict(temp_pca_features)[0]
#
#     axs[4, i].imshow(reconstruction)
#     axs[4, i].get_xaxis().set_visible(False)
#     axs[4, i].get_yaxis().set_visible(False)
#
#
#
# axs[0,2].set_title("Varying VAE Feature " + str(chosen_feature) + "                              ")
# axs[1,2].set_title("Varying VAE Feature " + str(chosen_feature_2) + "                              ")
# axs[2,2].set_title("Varying VAE Feature " + str(chosen_feature) + " and " + str(chosen_feature_2) + "                       ")
# axs[4,2].set_title("Varying PCA Feature " + str(chosen_pca_feature) + "                               ")
#
#
# # plt.savefig("Variational Eagle/Plots/transition_plot_individually_normalised_vae_vs_pca_" + str(encoding_dim) + "_features_" + str(run), bbox_inches='tight')
# plt.show()



