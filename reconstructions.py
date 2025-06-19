import os

from planar_flow_vae_eagle import reconstruction_indices

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
import time


tf.keras.mixed_precision.set_global_policy('float32')


tfb = tfp.bijectors
tfd = tfp.distributions






run = 3
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32


# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"





# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T




# load the images as a balanced dataset (D/T)

# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# get a list of all the ids of the galaxies
chosen_galaxies = list(all_properties["GalaxyID"])


# list to contain all galaxy images
all_images = []

# loop through each galaxy
for i, galaxy in enumerate(chosen_galaxies):

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

    # normalise the image (each band independently)
    image = normalise_independently(image)

    # add the image to the dataset
    all_images.append(image)

print("Original Dataset", len(all_images))

# split the data into training and testing data (200 images used for testing)
train_images = all_images
# train_images = all_images[:-200]
# test_images = np.array(all_images[-200:])

# print("Training Set", len(train_images))
# print("Testing Set", len(test_images))
# print()



# load the filenames of the augmented elliptical images
augmented_galaxies =  os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")

print("Augmented Ellipticals", len(augmented_galaxies))

for galaxy in augmented_galaxies:

    # load each augmented image
    image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)



# load the filenames of the augmented transitional images
augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")

print("Augmented Transitional", len(augmented_galaxies))

for galaxy in augmented_galaxies:

    # load each augmented image
    image = mpimg.imread("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)

# convert the training set to a numpy array
train_images = np.array(train_images)


print("Training Set", train_images.shape)
# print("Testing Set", test_images.shape)
print()










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

        # perform reparameterisation trick
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # initialise as a tensor of batch size shape (same shape as first latent feature)
        sum_log_det_jacobian = tf.zeros_like(z_mean[:, 0])

        # apply flow transformations
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
    z = tf.clip_by_value(z, -4 + 1e-4, 4 - 1e-4)

    sum_log_det_jacobian = 0.0

    # apply the flows
    for flow in flows:
        z, log_det = flow(z)
        sum_log_det_jacobian += log_det

    return z, sum_log_det_jacobian





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



# latent features vs pca

# load the weights
vae.load_weights("Variational Eagle/Weights/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")

# get the extracted features
extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")





# pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
# pca_top = PCA(n_components=4, svd_solver="full").fit(extracted_features)
#
#
# # number of images to reconstruct
# n = 12
#
#
# # get the images to reconstruct
# random.seed(5)
# reconstruction_indices = random.sample(range(train_images.shape[0]), n)
# extracted_features = extracted_features[reconstruction_indices]
#
#
# pca_features = pca.transform(extracted_features)
# pca_features = pca.inverse_transform(pca_features)
#
# pca_features_top = pca_top.transform(extracted_features)
# pca_features_top = pca_top.inverse_transform(pca_features_top)
#
# reconstructions = vae.decoder.predict(extracted_features)
# pca_reconstructions = vae.decoder.predict(pca_features)
# pca_reconstructions_top = vae.decoder.predict(pca_features_top)
#
# original_images = train_images[reconstruction_indices]
#
# fig, axs = plt.subplots(4, n, figsize=(n*10, 40))
#
# for i in range(0, n):
#
#     axs[0][i].imshow(original_images[i])
#     axs[0][i].set_aspect("auto")
#     axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     axs[1][i].imshow(reconstructions[i])
#     axs[1][i].set_aspect("auto")
#     axs[1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     axs[2][i].imshow(pca_reconstructions[i])
#     axs[2][i].set_aspect("auto")
#     axs[2][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     axs[3][i].imshow(pca_reconstructions_top[i])
#     axs[3][i].set_aspect("auto")
#     axs[3][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/latent_vs_pca_vs_top", bbox_inches="tight")
# plt.show()









# number of images to reconstruct
n = 12

# get the images to reconstruct
random.seed(5)
# reconstruction_indices = random.sample(range(train_images.shape[0]), n)
reconstruction_indices = [3165, 3108, 1172]
print(reconstruction_indices)
print(extracted_features.shape)
extracted_features_reconstruct = extracted_features[reconstruction_indices]



original_images = train_images[reconstruction_indices]


fig, axs = plt.subplots(11, n)

for i in range(0, n):

    axs[0][i].imshow(original_images[i])
    axs[0][i].set_aspect("auto")
    axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    for j in range(10, 1, -1):

        pca = PCA(n_components=j, svd_solver="full").fit(extracted_features)

        pca_features = pca.transform(extracted_features_reconstruct[j])
        pca_features = pca.inverse_transform(pca_features)

        pca_reconstruction = vae.decoder.predict(pca_features)

        axs[j+1][i].imshow(pca_reconstruction)
        axs[j+1][i].set_aspect("auto")
        axs[j+1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

plt.savefig("Variational Eagle/Plots/reconstruction_optimal_pca_features")
plt.show()
plt.close()







# flows vs normal reconstruction

# # get the images to reconstruct
# random.seed(5)
# reconstruction_indices = random.sample(range(train_images.shape[0]), n)
#
# # original images
# original_images = train_images[reconstruction_indices]
#
#
#
# # normal model reconstructions
# n_flows = 0
#
# # load the weights
# vae.load_weights("Variational Eagle/Weights/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")
#
# # get the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
# extracted_features = extracted_features[reconstruction_indices]
#
# # reconstruct the images
# reconstructions_normal = vae.decoder.predict(extracted_features)
#
#
#
# # flow model reconstructions
# n_flows = 3
#
# # load the weights
# vae.load_weights("Variational Eagle/Weights/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.weights.h5")
#
# # get the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
# extracted_features = extracted_features[reconstruction_indices]
#
# # reconstruct the images
# reconstructions_flows = vae.decoder.predict(extracted_features)
#
#
# # number of images to reconstruct
# n = 12
#
# fig, axs = plt.subplots(3, n, figsize=(n*10, 30))
#
# for i in range(0, n):
#
#     axs[0][i].imshow(original_images[i])
#     axs[0][i].set_aspect("auto")
#     axs[0][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     axs[1][i].imshow(reconstructions_normal[i])
#     axs[1][i].set_aspect("auto")
#     axs[1][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#     axs[2][i].imshow(reconstructions_flows[i])
#     axs[2][i].set_aspect("auto")
#     axs[2][i].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/flows_vs_normal", bbox_inches="tight")
# plt.show()







