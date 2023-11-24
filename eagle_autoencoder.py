import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# select to use GPU 0 on cosma
os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU


# returns a numpy array of the images to train the model
def get_images():

    # stores an empty list to contain all the image data to train the model
    train_images = []

    # loop through the directory containing all the image files
    for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):

        # open the fits file and get the image data (this is a numpy array of each pixel value)
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

        # append the image data to the main list containing all data of all the images
        train_images.append(image)

    # return this list
    return train_images


# get the images and labels to train the model
train_images = get_images()


# find the number of images that you will test the model on
testing_count = int(len(train_images)/4)

# split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
train_images = np.array(train_images[:testing_count*3])
test_images = np.array(train_images[testing_count:])


# # Instantiate a Keras tensor to allow us to build the model
# input_image = keras.Input(shape=(256, 256, 3))
#
# # layers for the encoder
# encoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(input_image)
#
# # layers for the decoder (extra one with 1 filter to get back to the correct shape)
# x = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(encoded)
# decoded = Conv2D(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)



# # Define keras tensor for the encoder
# input_image = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)
#
# # layers for the encoder
# x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 32)
# x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)                      # (64, 64, 16)
# x = Flatten()(x)                                                                                            # (65536) = (64 * 64 * 16)
# # x = Dense(units=16384)(x)
# x = Dense(units=1028)(x)
# # x = Dense(units=256)(x)
# # x = Dense(units=32)(x)
# encoded = Dense(units=32, activation="relu", name="z_mean")(x)                                                # (2)
#
# # build the encoder
# # encoder = keras.Model(inputs=input_image, outputs=encoded, name="encoder")
#
# # # Define keras tensor for the decoder
# # input_image_decoder = keras.Input(shape=(2))                                                                # (2)
#
# # layers for the decoder
# # x = Dense(units=32)(encoded)
# # x = Dense(units=256)(x)
# # x = Dense(units=1028)(encoded)
# x = Dense(units=16384)(encoded)
# # x = Dense(units=64*64*32, activation="relu")(x)                                           # (131072) = (64 * 64 * 32)
# x = Dense(units=64*64*16, activation="relu")(x)                                           # (65536) = (64 * 64 * 16)
# x = Reshape((64, 64, 16))(x)                                                                                # (64, 64, 16)
# x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (128, 128, 64)
# x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (265, 256, 32)
# decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)                   # (256, 256, 3)



# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=32)(x)                                                                              # (32)
encoded = Dense(units=2, name="encoded")(x)                                                         # (2)


# layers for the decoder
x = Dense(units=32)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)


# crate autoencoder
autoencoder = keras.Model(input_image, decoded)


# extract encoder layer and decoder layer from autoencoder
encoder_layer = autoencoder.get_layer("encoded")
decoder_layer = autoencoder.get_layer("decoded")

# define encoded input
encoded_input = keras.Input(shape=(2))


# crete encoder
encoder = keras.Model(autoencoder.input, encoder_layer.output)

# create decoder
decoder = keras.Model(encoder_layer.output, decoder_layer.output)
decoder.summary()


# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# train the model
# model_data = autoencoder.fit(train_images, train_images, epochs=3, batch_size=1, validation_data=(test_images, test_images))



# # number of galaxies on each side
# n = 15
#
# # size of each image
# image_size = 256
#
# # create the figure to store the images
# figure = np.zeros((image_size * n, image_size * n))
#
# # sample points within [-15, 15] standard deviations
# grid_x = np.linspace(-8, 8, n)
# grid_y = np.linspace(-8, 8, n)
#
# # populate each point on the figure
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#
#         z_sample = np.array([[xi, yi]])
#
#         x_decoded = decoder.predict(z_sample)
#
#         image = x_decoded[0].reshape(image_size, image_size, 3)
#
#         # add image to the figure
#         figure[i * image_size: (i+1) * image_size,
#                j * image_size: (j+1) * image_size] = image
#
# plt.figure(figsize=(20, 20))
# plt.imshow(figure)



# # create a subset of the validation data to reconstruct (first 10 images)
# images_to_reconstruct = test_images[:10]
#
# # number of images to reconstruct
# n = 10
#
# # reconstruct the images
# reconstructed_images = autoencoder.predict(test_images[:n])
#
# # create figure to hold subplots
# # fig, axs = plt.subplots(4, n-1, figsize=(20,8))
# fig, axs = plt.subplots(2, n-1, figsize=(20,4))
#
# # plot each subplot
# for i in range(0, n-1):
#
#     # show the original image (remove axes)
#     axs[0,i].imshow(test_images[i])
#     axs[0,i].get_xaxis().set_visible(False)
#     axs[0,i].get_yaxis().set_visible(False)
#
#     # show the reconstructed image (remove axes)
#     axs[1,i].imshow(reconstructed_images[i])
#     axs[1,i].get_xaxis().set_visible(False)
#     axs[1,i].get_yaxis().set_visible(False)
#
#     # # calculate residue (difference between two images) and show this
#     # residue_image = np.absolute(np.subtract(reconstructed_images[i], test_images[i]))
#     # axs[2,i].imshow(residue_image)
#     # axs[2,i].get_xaxis().set_visible(False)
#     # axs[2,i].get_yaxis().set_visible(False)
#     #
#     # # add an exponential transform to the residue to show differences more clearly
#     # exponential_residue = np.exp(5 * residue_image) - 1
#     # axs[3,i].imshow(exponential_residue)
#     # axs[3,i].get_xaxis().set_visible(False)
#     # axs[3,i].get_yaxis().set_visible(False)



# # build the encoder for feature extraction
# encoder = keras.Model(input_image, encoded)
# extracted_features = encoder.predict(train_images)
#
#
# print(extracted_features.tolist())
#
# # lists to store the values of each image for each extracted feature
# f1 = []
# f2 = []
# f3 = []
#
# # loop through each pair of values for each image and add the values to the individual lists
# for i in range(extracted_features.shape[0]):
#     f1.append(extracted_features[i][0])
#     f2.append(extracted_features[i][1])
#     f3.append(extracted_features[i][2])
#
#
#
# # linear regression via least squares between each of the 3 features
# b_12, a_12 = np.polyfit(f1, f2, deg=1)
# b_13, a_13 = np.polyfit(f1, f3, deg=1)
# b_23, a_23 = np.polyfit(f2, f3, deg=1)
#
#
# # Create sequence of 100 numbers from the minimum feature 1 value to the maximum feature 1 value (for regression line)
# sequence_f1 = np.linspace(np.min(f1), np.max(f1), num=100)
# sequence_f2 = np.linspace(np.min(f2), np.max(f2), num=100)
#
#
#
# # create the figure for the plot
# fig, axs = plt.subplots(2, 3, figsize=(25, 10))
#
# # pplot feature 1
# axs[0][0].hist(f1, bins=40)
# axs[0][0].set_title("Feature 1")
#
# # plot feature 2
# axs[0][1].hist(f2, bins=40)
# axs[0][1].set_title("Feature 2")
#
# # plot feature 3
# axs[0][2].hist(f3, bins=40)
# axs[0][2].set_title("Feature 3")
#
# # correlation between 1 and 2
# axs[1][0].scatter(f1, f2, s=5)
# axs[1][0].plot(sequence_f1, a_12 + b_12 * sequence_f1, color="k", lw=2)
# axs[1][0].set_title("Feature 2 Against Feature 1")
# axs[1][0].set_ylabel("Feature 2")
# axs[1][0].set_xlabel("Feature 1")
#
# # correlation between 1 and 3
# axs[1][1].scatter(f1, f3, s=5)
# axs[1][1].plot(sequence_f1, a_13 + b_13 * sequence_f1, color="k", lw=2)
# axs[1][1].set_title("Feature 3 Against Feature 1")
# axs[1][1].set_ylabel("Feature 3")
# axs[1][1].set_xlabel("Feature 1")
#
# # correlation between 2 and 3
# axs[1][2].scatter(f2, f3, s=5)
# axs[1][2].plot(sequence_f2, a_23 + b_23 * sequence_f2, color="k", lw=2)
# axs[1][2].set_title("Feature 3 Against Feature 2")
# axs[1][2].set_ylabel("Feature 3")
# axs[1][2].set_xlabel("Feature 2")







# plt.plot(model_data.history["loss"], label="training data")
# plt.plot(model_data.history["val_loss"], label="validation data")
# plt.legend()



plt.savefig("Plots/2_feature_latent_space")
plt.show()
