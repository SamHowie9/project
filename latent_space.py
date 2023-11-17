from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# select to use GPU 0 on cosma
os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU


# stores an empty list to contain all the image data to train the model
all_images = []

# loop through the directory containing all the image files
for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):

    # open the fits file and get the image data (this is a numpy array of each pixel value)
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

    # append the image data to the main list containing all data of all the images
    all_images.append(image)

# convert the list of images into a numpy array
all_images = np.array(all_images)

image = all_images[0]
print(image.shape)


# Define keras tensor for the encoder
input_image_encoder = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image_encoder)    # (128, 128, 32)
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)                      # (64, 64, 64)
x = Flatten()(x)                                                                                            # (262144) = (64 * 64 * 64)
encoded = Dense(units=2, activation="relu", name="z_mean")(x)                                                # (2)
# z_log_var = Dense(units=2, activation="relu", name="z_log_var")(x)                                          # (2)
# z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image_encoder, [z_mean, z_log_var, z], name="encoder")


# Define keras tensor for the decoder
input_image_decoder = keras.Input(shape=(2))                                                                # (2)

# layers for the decoder
x = Dense(units=64*64*32, activation="relu")(input_image_decoder)                                           # (131072) = (64 * 64 * 32)
x = Reshape((64, 64, 32))(x)                                                                                # (64, 64, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (128, 128, 64)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)             # (265, 256, 32)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same")(x)                   # (256, 256, 3)

# build the decoder
decoder = keras.Model(input_image, decoded, name="decoder")


# print(encoder(image))