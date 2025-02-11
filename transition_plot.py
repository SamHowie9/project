import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random




encoding_dim = 10
run = 2
epochs = 750
batch_size = 32




# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T




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

            # run image through encoder (get latent representation)
            z_mean, z_log_var, z = self.encoder(data)

            # form reconstruction (run latent representation through decoder)
            reconstruction = self.decoder(z)

            # binary cross entropy reconstruction loss
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )

            # root mean squared error reconstruction loss
            # reconstruction_loss = root_mean_squared_error(data, reconstruction)
            # reconstruction_loss = ops.sqrt(ops.mean(ops.sum(ops.square(data - reconstruction), axis=(1, 2, 3))))

            # get the kl divergence (mean for each extracted feature)
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))

            # loss is the sum of reconstruction loss and kl divergence
            total_loss = reconstruction_loss + kl_loss

        # gradient decent based on loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
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
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon





# encoding_dim = 20




# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 32)
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 64)
x = Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (32, 32, 128)
x = Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (16, 16, 256)
x = Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (8, 8, 512)
# x = Flatten()(x)                                                                                              # (8*8*512 = 32768)
x = GlobalAveragePooling2D()(x)                                                                                 # (512)
x = Dense(128, activation="relu")(x)                                                                            # (128)

z_mean = Dense(encoding_dim, name="z_mean")(x)
z_log_var = Dense(encoding_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# Define keras tensor for the decoder
latent_input = keras.Input(shape=(encoding_dim,))

# layers for the decoder
x = Dense(units=128, activation="relu")(latent_input)                                                           # (64)
x = Dense(units=512, activation="relu")(x)                                                                      # (256)
x = Dense(units=8*8*512, activation="relu")(x)                                                                  # (8*8*512 = 32768)
x = Reshape((8, 8, 512))(x)                                                                                     # (8, 8, 512)
x = Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (16, 16, 256)
x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (32, 32, 128)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 64)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
decoded = Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)        # (256, 256, 3)









# build the decoder
decoder = keras.Model(latent_input, decoded, name="decoder")
decoder.summary()


# build and compile the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())


# # load the weights
# vae.load_weights("Variational Eagle/Weights/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_" + str(run) + ".weights.h5")
# vae.load_weights("Variational Eagle/Weights/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")












# balanced dataset

# load the weights
# vae.load_weights("Variational Eagle/Weights/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_" + str(run) + ".weights.h5")
vae.load_weights("Variational Eagle/Weights/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")


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
extracted_features_switch = extracted_features.T

# perform pca on the extracted features
pca = PCA(n_components=5).fit(extracted_features)
extracted_features = pca.transform(extracted_features)
# extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T









# spirals only

# # load the weights
# vae.load_weights("Variational Eagle/Weights/Spirals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")
#
#
# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
#
# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # take only the sprial galaxies
# all_properties = all_properties[all_properties["n_r"] <= 2.5]
#
# # account for the training data in the dataframe
# all_properties = all_properties.iloc[:-200]
#
#
# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# encoding_dim = extracted_features.shape[1]
# extracted_features_switch = extracted_features.T
#
# # perform pca on the extracted features
# pca = PCA(n_components=5).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T








# ellipticals only

# # load the weights
# vae.load_weights("Variational Eagle/Weights/Ellipticals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")
#
#
# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
#
# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # # take only the sprial galaxies
# all_properties = all_properties[all_properties["n_r"] >= 4]
#
# # account for the training data in the dataframe
# all_properties = all_properties.iloc[:-12]
#
# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Ellipticals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# encoding_dim = extracted_features.shape[1]
# extracted_features_switch = extracted_features.T
#
# # perform pca on the extracted features
# pca = PCA(n_components=5).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T






# print(extracted_features.shape)







num_varying_features = 13


med_pca_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
print(len(med_pca_features))



# transition plot for all extracted features

# fig, axs = plt.subplots(len(extracted_features.T), num_varying_features, figsize=(15, 25))
#
# for i in range(len(extracted_features.T)):
#
#     varying_feature_values = np.linspace(np.min(extracted_features.T[i]), np.max(extracted_features.T[i]), num_varying_features)
#
#     for j in range(num_varying_features):
#
#         temp_pca_features = med_pca_features.copy()
#         temp_pca_features[i] = varying_feature_values[j]
#
#         temp_features = temp_pca_features
#         temp_features = np.expand_dims(temp_features, axis=0)
#
#         temp_features = pca.inverse_transform(temp_pca_features)
#         temp_features = np.expand_dims(temp_features, axis=0)
#
#         reconstruction = vae.decoder.predict(temp_features)[0]
#
#         axs[i][j].imshow(reconstruction)
#         axs[i][j].get_xaxis().set_visible(False)
#         axs[i][j].get_yaxis().set_visible(False)
#
# plt.savefig("Variational Eagle/Plots/new_transition_plot_vae_pca_all_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()



# transition plot for specific extracted feature

# chosen_feature = 2
#
# varying_feature_values = np.linspace(np.min(extracted_features.T[chosen_feature]), np.max(extracted_features.T[chosen_feature]), num_varying_features)
#
# fig, axs = plt.subplots(1, num_varying_features, figsize=(num_varying_features, 3))
#
# for i in range(num_varying_features):
#     temp_pca_features = med_pca_features.copy()
#     temp_pca_features[chosen_feature] = varying_feature_values[i]
#
#     temp_features = temp_pca_features
#     temp_features = np.expand_dims(temp_features, axis=0)
#
#     temp_features = pca.inverse_transform(temp_pca_features)
#     temp_features = np.expand_dims(temp_features, axis=0)
#
#     reconstruction = vae.decoder.predict(temp_features)[0]
#
#     axs[i].imshow(reconstruction)
#     # axs[i].get_xaxis().set_visible(False)
#     # axs[i].get_yaxis().set_visible(False)
#
#     axs[i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     # axs[i].set_title(round(varying_feature_values[i], 2))
#     axs[i].set_xlabel(round(varying_feature_values[i], 2))
#
#
# # fig.suptitle("PCA Feature 0")
# supxlabel = fig.supxlabel("PCA Feature 2")
# supxlabel.set_y(0.2)
#
# plt.savefig("Variational Eagle/Plots/transition_plot_vae_pca_feature_2", bbox_inches='tight')
# plt.show()




# transition plot for group of features

chosen_features = [0, 1, 2, 3, 4]

fig, axs = plt.subplots(len(chosen_features), num_varying_features, figsize=(num_varying_features, 8))

for i, feature in enumerate(chosen_features):

    varying_feature_values = np.linspace(np.min(extracted_features.T[i]), np.max(extracted_features.T[i]), num_varying_features)

    for j in range(num_varying_features):

        temp_pca_features = med_pca_features.copy()
        temp_pca_features[feature] = varying_feature_values[j]

        temp_features = temp_pca_features
        temp_features = np.expand_dims(temp_features, axis=0)

        temp_features = pca.inverse_transform(temp_pca_features)
        temp_features = np.expand_dims(temp_features, axis=0)

        reconstruction = vae.decoder.predict(temp_features)[0]

        axs[i][j].imshow(reconstruction)
        axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        axs[i][j].set_xlabel(round(varying_feature_values[j], 2))

        if j == (num_varying_features - 1)/2:
            axs[i][j].set_xlabel(str(round(varying_feature_values[j], 2)) + "\nPCA Feature " + str(feature))

plt.savefig("Variational Eagle/Transition Plots/Balanced/" + str(encoding_dim) + "_features_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_" + str(run) + "_" + str(num_varying_features) + "_images", bbox_inches='tight')
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



