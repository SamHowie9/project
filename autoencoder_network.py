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


# create a figure to store the two different plots
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,6))
#
# # set the title, labels and accuracy axis limits of the training data plot
# ax1.set_title("Training Loss")
# ax1.set_ylabel("Loss")
# ax1.set_xlabel("Epochs")
#
# # set the title, labels and accuracy axis limits of the validation data plot
# ax2.set_title("Validation Loss")
# ax2.set_ylabel("Loss")
# ax2.set_xlabel("Epochs")


# Instantiate a Keras tensor to allow us to build the model
input_image = keras.Input(shape=(50, 50, 1))

# layers for the encoder
encoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(input_image)
# encoded = MaxPooling2D(pool_size=2, padding="same")(x)

# layers for the decoder
decoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(encoded)
# decoded = UpSampling2D(size=2)(x)


# encoded = layers.Dense(25, activation="relu")(input_image)
# decoded = layers.Dense(1, activation="sigmoid")(encoded)


# create and compile the autoencoder model
autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print(train_images.shape)
print(test_images.shape)
print()


# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=50, batch_size=1, validation_data=(test_images, test_images))



# # plot the loss for the training and validation data
# training_loss, = plt.plot(model_data.history["loss"], label="Training Data")
# validation_loss, = plt.plot(model_data.history["val_loss"], label="Validation Data")
#
# # set the axis titles and legend for the training and validation loss plot
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend(loc="upper right")


# create a subset of the validation data to reconstruct (first 10 images)
images_to_reconstruct = test_images[:10]
images_to_reconstruct = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(encoded)

# number of images to reconstruct
n = 10

# reconstruct the images
reconstructed_images = autoencoder.predict(test_images[:n])
reconstructed_images = np.array(Conv2D(filters=1, kernel_size=3, activation="relu", padding="same")(reconstructed_images))
# reconstructed_images = UpSampling2D(size=2)(x)

print(test_images[0].shape)
print(reconstructed_images[0].shape)


plt.figure(figsize=(20,4))
for i in range(1, n):

    # display the original images (with no axes)
    ax_o = plt.subplot(2, n, i)
    plt.imshow(test_images[i].reshape(50, 50))
    ax_o.get_xaxis().set_visible(False)
    ax_o.get_yaxis().set_visible(False)

    # display the reconstructed images (with no axes)
    ax_r = plt.subplot(2, n, i + n)
    plt.imshow(reconstructed_images[i].reshape(50, 50))

    ax_r.get_xaxis().set_visible(False)
    ax_r.get_yaxis().set_visible(False)



plt.show()
plt.savefig("Plots/autoencoder_conv2d_reconstruction")

