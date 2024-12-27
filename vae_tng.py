import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose
import numpy as np
import pandas as pd
import random
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg



# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tf.config.list_physical_devices('GPU')


encoding_dim = 25

run = 3

# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"

# number of epochs for run
epochs = 300


# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T

# normalise each band to r
def normalise_to_r(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[1]) - np.min(image[1]))
    return image.T

# convert fwhm value to sigma
def fwhm_to_sigma(fwhm):
    return fwhm/(2 * np.sqrt(2 * np.log(2)))

# image processing for each image
def image_processing(image):

    # take only the g,r,i bands (ignore z)
    image = image[0:3]

    # apply gaussian filter to each band
    image[0] = gaussian_filter(image[0], sigma=fwhm_to_sigma(1.5))
    image[1] = gaussian_filter(image[1], sigma=fwhm_to_sigma(1.5))
    image[2] = gaussian_filter(image[2], sigma=fwhm_to_sigma(2))

    # random seed for reproducibility
    random.seed(1)

    # add random gaussian noise to each band of the image
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image[0]), len(image[0])))
        image[i] = image[i] + gaussian

    # convert image to numpy array of type float32 (for the cv2 resizing function to work)
    image = np.array(image).astype(np.float32)

    # image resizing (enlarging and shrinking use different interpolation algorithms for the best results
    if len(image[0] < 256):
        # enlarge (stretch) the image to 256x256 with bicubic interpolation (best for enlarging images although slower than bilinear)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_CUBIC)
    else:
        # shrink the image to 256x256 using area interpolation (best for shrinking images)
        image = cv2.resize(image.T, (256, 256), interpolation=cv2.INTER_AREA)

    #transpose for normalisation
    image = image.T

    # normalise each band individually
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i] - np.min(image[i])))

    # return the new image
    return image


# list to contain all galaxy images
all_images = []

# list of all the galaxy images to load
galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/TNG Images/")

# loop through each image
for galaxy in galaxies:

    # open the fits file and get the image
    hdu_list = fits.open("/cosma7/data/durham/dc-howi1/project/TNG Images/" + galaxy)
    image = hdu_list[0].data

    # perform the image processing
    image = image_processing(image)

    # add the transpose of the image to the list
    all_images.append(image.T)


print(np.array(all_images[0]).shape)
print(np.array(all_images).shape)











# # Define VAE model with custom time step
# class VAE(keras.Model):
#
#     def __init__(self, encoder, decoder, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#         self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
#
#     @property
#     def metrics(self):
#         return[
#             self.total_loss_tracker,
#             self.reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#         ]
#
#     # custom train step
#     def train_step(self, data):
#
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = ops.mean(
#                 ops.sum(
#                     keras.losses.binary_crossentropy(data, reconstruction),
#                     axis=(1, 2),
#                 )
#             )
#             kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
#             kl_loss = ops.mean(ops.sum(kl_loss + kl_loss))
#             total_loss = reconstruction_loss + kl_loss
#
#         grads = tape.gradient(total_loss, self.trainable_weights)
#
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         self.kl_loss_tracker.update_state(kl_loss)
#
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }
#
#
#
#
#
#
# # define sampling layer
# class Sampling(Layer):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.seed_generator = keras.random.SeedGenerator(1337)
#
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = ops.shape(z_mean)[0]
#         dim = ops.shape(z_mean)[1]
#         epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
#         return z_mean + ops.exp(0.5 * z_log_var) * epsilon
#
#
#
#
#
# # all_rmse_train = []
# # all_rmse_test = []
# #
# # for encoding_dim in range(1, 51):
#
# # number of extracted features
# # encoding_dim = 32
#
# # Define keras tensor for the encoder
# input_image = keras.Input(shape=(256, 256, 3))                                                                  # (256, 256, 3)
#
# # layers for the encoder
# x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)                # (128, 128, 64)
# x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (64, 64, 32)
# x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                          # (32, 32, 16)
# x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (16, 16, 8)
# x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                           # (8, 8, 4)
# x = Flatten()(x)                                                                                                # (256)
# x = Dense(units=64)(x)                                                                                          # (64)
# z_mean = Dense(encoding_dim, name="z_mean")(x)
# z_log_var = Dense(encoding_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
#
# # build the encoder
# encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
# encoder.summary()
#
#
# # Define keras tensor for the decoder
# latent_input = keras.Input(shape=(encoding_dim,))
#
# # layers for the decoder
# x = Dense(units=64)(latent_input)                                                                               # (64)
# x = Dense(units=256)(x)                                                                                         # (256)
# x = Reshape((8, 8, 4))(x)                                                                                       # (8, 8, 4)
# x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (16, 16, 4)
# x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)                  # (32, 32, 8)
# x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (64, 64, 16)
# x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (128, 128, 32)
# x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                 # (256, 256, 64)
# # decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)    # (128, 128, 3)
# decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same", name="decoded")(x)       # (128, 128, 3)
#
# # build the decoder
# decoder = keras.Model(latent_input, decoded, name="decoder")
# decoder.summary()
#
#
#
#
# # build and compile the VAE
# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam())
#
#
#
#
# # train the model
# model_loss = vae.fit(train_images, epochs=epochs, batch_size=1)
#
# # or load the weights from a previous run
# # vae.load_weights("Variational Eagle/Weights/Normalised to r/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_1.weights.h5")
#
#
# # save the weights
# vae.save_weights(filepath="Variational Eagle/Weights/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_" + str(run) + ".weights.h5", overwrite=True)
#
# # generate extracted features from trained encoder and save as numpy array
# extracted_features = vae.encoder.predict(train_images)
# np.save("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_" + str(run) + ".npy", extracted_features)
#
# print(np.array(extracted_features).shape)
#
# # get loss, reconstruction loss and kl loss and save as numpy array
# loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
# print("\n \n" + str(encoding_dim))
# print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
# np.save("Variational Eagle/Loss/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_loss_" + str(run) + ".npy", loss)
#
#
#
#
#
#
# # loss plot for individual run
# fig, axs1 = plt.subplots()
# axs1.plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss")
# axs1.set_ylabel("reconstruction loss")
# axs2 = axs1.twinx()
# axs2.plot(model_loss.history["kl_loss"], label="KL Loss", color="y")
# axs2.set_ylabel("KL Loss")
# plt.legend()
#
# plt.savefig("Variational Eagle/Plots/fully_balanced_" + str(encoding_dim) + "_feature_loss_" + str(run))
# plt.show()
#
#
#
#
#
#
# # Form reconstructions
#
# # number of images to reconstruct
# n = 12
#
# # create a subset of the validation data to reconstruct (first 10 images)
# images_to_reconstruct = test_images[:n]
# # images_to_reconstruct = train_images[n:]
#
# # print(images_to_reconstruct.shape)
#
# # reconstruct the images
# test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
# reconstructed_images = vae.decoder.predict(test_features)
#
# # create figure to hold subplots
# fig, axs = plt.subplots(2, n-1, figsize=(18,5))
#
# # plot each subplot
# for i in range(0, n-1):
#
#     original_image = normalise_independently(images_to_reconstruct[i])
#     reconstructed_image = normalise_independently(reconstructed_images[i])
#
#     # show the original image (remove axes)
#     # axs[0,i].imshow(images_to_reconstruct[i])
#     axs[0,i].imshow(original_image)
#     axs[0,i].get_xaxis().set_visible(False)
#     axs[0,i].get_yaxis().set_visible(False)
#
#     # show the reconstructed image (remove axes)
#     # axs[1,i].imshow(reconstructed_images[i])
#     axs[1,i].imshow(reconstructed_image)
#     axs[1,i].get_xaxis().set_visible(False)
#     axs[1,i].get_yaxis().set_visible(False)
#
# plt.savefig("Variational Eagle/Reconstructions/Validation/balanced_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_reconstruction_" + str(run))
# plt.show()








#     # find rmse (training and validation)
#
#     vae.load_weights("Variational Eagle/Weights/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights_3.weights.h5")
#     features_train = np.load("Variational Eagle/Extracted Features/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_3.npy")
#     # reconstructions_train = vae.decoder.predict(features_train[0])
#
#     rmse_train = []
#     rmse_test = []
#
#     # for i in range(len(train_images)):
#     #     squared_diff = (train_images[i] - reconstructions_train[i]) ** 2
#     #     rmse = np.sqrt(np.mean(squared_diff))
#     #
#     #     rmse_train.append(rmse)
#     #
#     # print(np.median(np.array(rmse_train)))
#     # all_rmse_train.append(np.median(np.array(rmse_train)))
#
#
#     features_test, _, _ = vae.encoder.predict(test_images)
#     reconstructions_test = vae.decoder.predict(features_test)
#
#     for i in range(len(test_images)):
#         squared_diff = (test_images[i] - reconstructions_test[i]) ** 2
#         rmse = np.sqrt(np.mean(squared_diff))
#
#         rmse_test.append(rmse)
#
#
#     all_rmse_train.append(np.median(np.array(rmse_train)))
#     all_rmse_test.append(np.median(np.array(rmse_test)))
#
#
# # np.save("Variational Eagle/Loss/rmse_train_3", all_rmse_train)
# np.save("Variational Eagle/Loss/rmse_test_3", all_rmse_test)
#
# # for i in range(len(test_images)):








# epochs = [25, 50, 100, 150, 200, 250, 300, 350, 400]
#
# for epoch in epochs:
#
#     # load the weights
#     vae.load_weights("Variational Eagle/Weights/" + str(encoding_dim) + "_feature_" + str(epoch) + "_epoch_weights.weights.h5")
#
#     # # save the weights
#     # vae.save_weights(filepath="Variational Eagle/Weights/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_weights.weights.h5", overwrite=True)
#     #
#     #
#     # # generate extracted features from trained encoder and save as numpy array
#     # extracted_features = vae.encoder.predict(train_images)
#     # np.save("Variational Eagle/Extracted Features/" + str(encoding_dim) + "feature_" + str(epochs) + "_epoch_features.npy", extracted_features)
#     #
#     #
#     # # get loss, reconstruction loss and kl loss and save as numpy array
#     # loss = np.array([model_loss.history["loss"][-1], model_loss.history["reconstruction_loss"][-1], model_loss.history["kl_loss"][-1]])
#     # print("\n \n" + str(encoding_dim))
#     # print(str(loss[0]) + "   " + str(loss[1]) + "   " + str(loss[2]) + "\n")
#     # np.save("Variational Eagle/Loss/" + str(encoding_dim) + "_feature" + str(epochs) + "_epoch_loss.npy", loss)
#
#
#
#
#     # # loss plot for individual run
#     # fig, axs1 = plt.subplots()
#     # axs1.plot(model_loss.history["reconstruction_loss"], label="Reconstruction Loss")
#     # axs1.set_ylabel("reconstruction loss")
#     # axs2 = axs1.twinx()
#     # axs2.plot(model_loss.history["kl_loss"], label="KL Loss", color="y")
#     # axs2.set_ylabel("KL Loss")
#     # plt.legend()
#     #
#     # plt.savefig("Variational Eagle/Plots/" + str(encoding_dim) + "_feature" + str(epochs) + "_epoch_loss")
#     # plt.show()
#
#
#
#
#     # number of images to reconstruct
#     n = 12
#
#     # create a subset of the validation data to reconstruct (first 10 images)
#     # images_to_reconstruct = test_images[n:]
#     images_to_reconstruct = train_images[n:]
#
#     # reconstruct the images
#     test_features, _, _ = vae.encoder.predict(images_to_reconstruct)
#     reconstructed_images = vae.decoder.predict(test_features)
#
#     # create figure to hold subplots
#     fig, axs = plt.subplots(2, n-1, figsize=(18,5))
#
#     # plot each subplot
#     for i in range(0, n-1):
#
#         # show the original image (remove axes)
#         axs[0,i].imshow(images_to_reconstruct[i])
#         axs[0,i].get_xaxis().set_visible(False)
#         axs[0,i].get_yaxis().set_visible(False)
#
#         # show the reconstructed image (remove axes)
#         axs[1,i].imshow(reconstructed_images[i])
#         axs[1,i].get_xaxis().set_visible(False)
#         axs[1,i].get_yaxis().set_visible(False)
#
#     plt.savefig("Variational Eagle/Reconstructions/Training/" + str(encoding_dim) + "_feature_" + str(epoch) + "_epoch_reconstruction_2")
#     plt.show()

