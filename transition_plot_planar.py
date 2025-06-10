import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, losses, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D, Layer, Input
from tensorflow.random import normal, Generator
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.decomposition import PCA




run = 1
encoding_dim = 30
n_flows = 3
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32


os.environ["CUDA_VISIBLE_DEVICES"]="1"







# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 1) |
#                           (all_properties["flag_r"] == 4) |
#                           (all_properties["flag_r"] == 5) |
#                           (all_properties["flag_r"] == 6))].index.tolist()
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # account for the testing dataset
# all_properties = all_properties.iloc[:-200]







# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T







# Define VAE model with custom train step
class VAE(Model):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

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
            z_mean, z_log_var, z, sum_log_det_jacobians  = self.encoder(data)

            print("Z Shape", z_mean.shape, z.shape)

            # form the reconstruction (run latent representation through decoder)
            reconstruction = self.decoder(z)

            # reconstruction loss
            reconstruction_loss = tf.reduce_mean(losses.binary_crossentropy(data, reconstruction))

            # kl loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = (tf.reduce_sum(kl_loss, axis=1) - sum_log_det_jacobians) / z.shape[1]
            kl_loss = tf.reduce_mean(kl_loss)

            # total loss
            # total_loss = reconstruction_loss + kl_loss
            total_loss = reconstruction_loss + (beta * kl_loss)



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

    def __init__(self, latent_dim, n_flows=1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.n_flows = n_flows
        self.flows = [PlanarFlow(latent_dim) for _ in range(n_flows)]

    def call(self, inputs):

        # get the latent distributions
        z_mean, z_log_var = inputs

        # find the batch size and number of latent features (dim)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # generate the random variables
        epsilon = tf.random.normal(shape=(batch, dim))

        # perform reparameterization trick
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

        # apply flow transformations
        sum_log_det_jacobian = 0.0
        for flow in self.flows:
            z, log_det = flow(z)
            sum_log_det_jacobian += log_det

        return z, sum_log_det_jacobian





# define planar flows
class PlanarFlow(Layer):

    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):

        # Initialize weights and bias for the planar transformation
        self.u = self.add_weight(shape=(self.latent_dim,), initializer='random_normal', trainable=True)
        self.w = self.add_weight(shape=(self.latent_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(), initializer='zeros', trainable=True)

    def call(self, z):

        # parameterization of u (ensure eTu > -1)
        u_hat = self.u + (tf.nn.softplus(tf.reduce_sum(self.w * self.u)) - 1 - tf.reduce_sum(self.w * self.u)) * self.w / (tf.norm(self.w) ** 2 + 1e-8)

        # transformation
        w_dot_z = tf.reduce_sum(self.w * z, axis=1, keepdims=True)
        activation = tf.tanh(w_dot_z + self.b)
        z_transformed = z + (u_hat * activation)

        # derivative of flow function
        psi = (1.0 - tf.square(activation)) * self.w

        # compute the log det jacobian
        det_jacobian = 1.0 + tf.reduce_sum(psi * u_hat, axis=1)  # shape: (batch_size,)
        log_det_jacobian = tf.math.log(tf.abs(det_jacobian) + 1e-8)  # add epsilon for numerical stability

        return z_transformed, log_det_jacobian





# apply the flows to the latent vectors after training
def apply_flows(z_mean, flows):

    # convert vectors to tensor and clip (as done in sampling layer)
    z = tf.convert_to_tensor(z_mean, dtype=tf.float32)
    z = tf.clip_by_value(z, -4+1e-4, 4-1e-4)

    # apply the flows
    for flow in flows:
        z, _ = flow(z)

    # return the transformed vector
    return z





# Define keras tensor for the encoder
input_image = Input(shape=(256, 256, 3))                                                                               # (256, 256, 3)

# layers for the encoder
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 32)
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 64)
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (32, 32, 128)
x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (16, 16, 256)
x = layers.Conv2D(filters=512, kernel_size=3, strides=2, activation="relu", padding="same")(x)                         # (8, 8, 512)
# x = Flatten()(x)                                                                                                     # (8*8*512 = 32768)
x = layers.GlobalAveragePooling2D()(x)                                                                                 # (512)
x = layers.Dense(128, activation="relu")(x)                                                                            # (128)

z_mean = layers.Dense(encoding_dim, name="z_mean")(x)
z_log_var = layers.Dense(encoding_dim, name="z_log_var")(x)

z, sum_log_det_jacobians = Sampling(encoding_dim, n_flows=n_flows)([z_mean, z_log_var])

# build the encoder
encoder = Model(input_image, [z_mean, z_log_var, z, sum_log_det_jacobians], name="encoder")
encoder.summary()



# Define keras tensor for the decoder
latent_input = Input(shape=(encoding_dim,))

# layers for the decoder
x = layers.Dense(units=128, activation="relu")(latent_input)                                                           # (64)
x = layers.Dense(units=512, activation="relu")(x)                                                                      # (256)
x = layers.Dense(units=8*8*512, activation="relu")(x)                                                                  # (8*8*512 = 32768)
x = layers.Reshape((8, 8, 512))(x)                                                                                     # (8, 8, 512)
x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (16, 16, 256)
x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation="relu", padding="same")(x)                # (32, 32, 128)
x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 64)
x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
decoded = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation="sigmoid", padding="same")(x)         # (256, 256, 3)

# build the decoder
decoder = Model(latent_input, decoded, name="decoder")
decoder.summary()



# build and compile the VAE
vae = VAE(encoder, decoder)
# vae.compile(optimizer=optimizers.Adam(clipnorm=1.0))
vae.compile(optimizer=optimizers.Adam())





vae.build(input_shape=(None, 256, 256, 3))

# or load the weights from a previous run
vae.load_weights("Variational Eagle/Weights/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")



# load the original and transformed features
z_mean = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
z_transformed = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

# remove the augmented image features
# z_mean = z_mean[:len(all_properties)]
# z_transformed = z_transformed[:len(all_properties)]



# perform PCA on both sets of features
pca_mean = PCA(n_components=0.999).fit(z_mean)
z_mean = pca_mean.transform(z_mean)
pca_transformed = PCA(n_components=0.999).fit(z_transformed)
z_transformed = pca_transformed.transform(z_transformed)





# select transformed or mean
# extracted_features = z_mean
extracted_features = z_transformed
pca = pca_transformed



# scale font on plots
default_size = plt.rcParams['font.size']
plt.rcParams.update({'font.size': default_size * 5})




# transition plot for group of features

num_varying_features = 13

med_pca_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
print(len(med_pca_features))

chosen_features = [0, 1, 2, 3]

fig, axs = plt.subplots(len(chosen_features), num_varying_features, figsize=(num_varying_features*5, len(chosen_features)*5))

for i, feature in enumerate(chosen_features):

    varying_feature_values = np.linspace(np.min(extracted_features.T[feature]), np.max(extracted_features.T[feature]), num_varying_features)

    for j in range(num_varying_features):

        temp_pca_features = med_pca_features.copy()
        temp_pca_features[feature] = varying_feature_values[j]

        temp_features = temp_pca_features
        temp_features = np.expand_dims(temp_features, axis=0)



        temp_features = pca_transformed.inverse_transform(temp_pca_features)
        temp_features = np.expand_dims(temp_features, axis=0)

        reconstruction = vae.decoder.predict(temp_features)[0]

        axs[i][j].imshow(reconstruction)

        axs[i][j].set_aspect("auto")

        axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


        # axs[i][j].set_xlabel(round(varying_feature_values[j], 2))
        #
        # if j == (num_varying_features - 1)/2:
        #     axs[i][j].set_xlabel(str(round(varying_feature_values[j], 2)) + "\nPCA Feature " + str(feature))

    axs[i][0].set_ylabel(feature, rotation=0, labelpad=40, va='center')

fig.text(0.09, 0.5, 'Extracted Features', va='center', rotation='vertical')

fig.subplots_adjust(wspace=0, hspace=0.05)

plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/pca_top_4_latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run), bbox_inches='tight')
plt.show()









# transition plot for all extracted features

num_varying_features = 13

med_pca_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
print(len(med_pca_features))

fig, axs = plt.subplots(len(extracted_features.T), num_varying_features, figsize=(num_varying_features*5, len(extracted_features.T)*5))

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

        axs[i][j].set_aspect("auto")

        # remove the ticks
        axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

        # remove the spines
        # for spine in axs[i][j].spines.values():
        #     spine.set_visible(False)

    axs[i][0].set_ylabel(i, rotation=0, labelpad=40, va='center')

fig.text(0.09, 0.5, 'Extracted Features', va='center', rotation='vertical')


fig.subplots_adjust(wspace=0, hspace=0.05)

plt.savefig("Variational Eagle/Transition Plots/Normalising Flow Balanced/pca_latent_" + str(encoding_dim) + "_flows_" + str(n_flows) + "_" + str(run), bbox_inches='tight')
plt.show()








# # # take just the spirals
# # disk_structures = np.array(all_properties["n_r"] <= 2.5)
# # extracted_features = extracted_features[disk_structures]
#
# # take just the ellipticals
# # disk_structures = np.array(all_properties["n_r"] >= 4)
# # extracted_features = extracted_features[disk_structures]
#
#
#
#
#
#
#
#
#
# # spirals only
#
# # # load the weights
# # vae.load_weights("Variational Eagle/Weights/Spirals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")
# #
# #
# # # load structural and physical properties into dataframes
# # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
# #
# #
# # # dataframe for all properties
# # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
# #
# #
# # # find all bad fit galaxies
# # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# #
# # # remove those galaxies
# # for galaxy in bad_fit:
# #     all_properties = all_properties.drop(galaxy, axis=0)
# #
# # # take only the sprial galaxies
# # all_properties = all_properties[all_properties["n_r"] <= 2.5]
# #
# # # account for the training data in the dataframe
# # all_properties = all_properties.iloc[:-200]
# #
# #
# # # load the extracted features
# # extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# # encoding_dim = extracted_features.shape[1]
# # extracted_features_switch = extracted_features.T
# #
# # # perform pca on the extracted features
# # pca = PCA(n_components=5).fit(extracted_features)
# # extracted_features = pca.transform(extracted_features)
# # extracted_features_switch = extracted_features.T
#
#
#
#
#
#
#
#
# # ellipticals only
#
# # # load the weights
# # vae.load_weights("Variational Eagle/Weights/Ellipticals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_weights_" + str(run) + ".weights.h5")
# #
# #
# # # load structural and physical properties into dataframes
# # structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# # physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
# #
# #
# # # dataframe for all properties
# # all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
# #
# #
# # # find all bad fit galaxies
# # bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# #
# # # remove those galaxies
# # for galaxy in bad_fit:
# #     all_properties = all_properties.drop(galaxy, axis=0)
# #
# # # # take only the sprial galaxies
# # all_properties = all_properties[all_properties["n_r"] >= 4]
# #
# # # account for the training data in the dataframe
# # all_properties = all_properties.iloc[:-12]
# #
# # # load the extracted features
# # extracted_features = np.load("Variational Eagle/Extracted Features/Ellipticals/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# # encoding_dim = extracted_features.shape[1]
# # extracted_features_switch = extracted_features.T
# #
# # # perform pca on the extracted features
# # pca = PCA(n_components=5).fit(extracted_features)
# # extracted_features = pca.transform(extracted_features)
# # extracted_features_switch = extracted_features.T

































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



