from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import keras
from keras import layers
from astropy.io import fits
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# select to use GPU 0 on cosma
os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU


# returns a numpy array of the images to train the model
def get_images():

    # stores an empty list to contain all the image data to train the model
    train_images = []

    # loop through the directory containing all the image files
    for file in os.listdir("Images"):
        # skip the file containing the label names (and other galaxy information)
        if file == "desY1stripe82_GZ1_ES.fits":
            continue

        # open the fits file and get the image data (this is a numpy array of each pixel value)
        hdu_list = fits.open("Images/" + file)
        image_data = hdu_list[0].data

        # append the image data to the main list containing all data of all the images
        train_images.append(image_data)

    # return this list
    return train_images


# returns a numpy array of the labels for all of the images
def get_labels():

    # open the file containing data about all images, including the type of each galaxy
    hdu_list = fits.open("Images/desY1stripe82_GZ1_ES.fits")

    # create a dataframe to store all of the data
    df = pd.DataFrame(hdu_list[1].data)

    # store the type of galaxy for each image (0 is spiral, 1 is elliptical)
    galaxy_types = df["ELLIPTICAL"].to_list()

    # return the list
    return galaxy_types


# get the images and labels to train the model
train_images = get_images()
train_labels = get_labels()

# find a quarter the number of images (and truncate this to ensure we have an integer value but the training and testing data share no common values)
half_images = int(len(train_images)/4)

# split the images and labels in into 3/4 for training and 1/4 testing, convert these lists into numpy arrays
test_images = np.array(train_images[half_images:])
test_labels = np.array(train_labels[half_images:])
train_images = np.array(train_images[:half_images*3])
train_labels = np.array(train_labels[:half_images*3])

# expand the dimensions of the images to add the number of colours (here we are greyscale so use 1 making the images of shape, (50, 50, 1)
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)


# list to store the training loss values
training_loss = []

# list to store the validation loss values
validation_loss = []

# list to store the number of filters
all_kernels = list(range(1, 10))

# set the number of epochs
epochs = 50

for kernel_size in all_kernels:

    print()
    print()
    print("Using a kernel size of ", kernel_size)
    print()

    # Instantiate a Keras tensor to allow us to build the model
    input_image = keras.Input(shape=(50, 50, 1))

    # layers for the encoder
    encoded = Conv2D(filters=8, kernel_size=kernel_size, activation="relu", padding="same")(input_image)

    # layers for the decoder (extra one with 1 filter to get back to the correct shape)
    decoded = Conv2D(filters=8, kernel_size=kernel_size, activation="relu", padding="same")(encoded)
    decoded = Conv2D(filters=1, kernel_size=kernel_size, activation="sigmoid", padding="same")(encoded)


    # create and compile the autoencoder model
    autoencoder = keras.Model(input_image, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


    # train the model
    model_data = autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=1, validation_data=(test_images, test_images))

    training_loss.append(model_data.history["loss"][epochs-1])
    validation_loss.append(model_data.history["val_loss"][epochs-1])


all_filters = np.array(all_filters)
training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)


fig, axs = plt.subplots(1, 3, figsize=(18,5))
axs[0].plot(all_filters, training_loss, label="Training Data")
axs[0].plot(all_filters, validation_loss, label="Validation Data", color="#ff7f0e")
axs[0].legend(loc="upper right")
axs[0].set_ylabel("Loss")
axs[0].set_xlabel("Kernel Size")
axs[1].plot(all_filters, training_loss, label="Training Data")
axs[1].set_xlabel("Kernel Size")
axs[1].legend(loc="upper right")
axs[2].plot(all_filters, validation_loss, label="Validation Data", color="#ff7f0e")
axs[2].set_xlabel("Kernel Size")
axs[2].legend(loc="upper right")


# plt.figure(figsize=(18,5))
#
# # plot the loss for the training and validation data
# ax = plt.subplot(1, 3, 1)
# training_loss, = plt.plot(all_filters, training_loss, label="Training Data")
# validation_loss, = plt.plot(all_filters, validation_loss, label="Validation Data", color="#ff7f0e")
# ax.legend(loc="upper left")
# ax.set_ylabel("Loss")
# ax.set_xlabel("Filters")
#
# # ax = plt.subplot(1, 3, 2)
# # training_loss_single = plt.plot(all_filters, training_loss, label="Training Data")
# # ax.legend(loc="upper left")
# # ax.set_xlabel("Filters")
# #
# ax = plt.subplot(1, 3, 3)
# validation_loss, = plt.plot(all_filters, validation_loss, label="Valiation Data", color="#ff7f0e")
# ax.legend(loc="upper left")
# ax.set_xlabel("Filters")

# set the axis titles and legend for the training and validation loss plot
# plt.ylabel("Loss")
# plt.xlabel("Filters")
# plt.legend(loc="upper right")

plt.show()
plt.savefig("Plots/autoencoder_conv2d_kernel_size")

