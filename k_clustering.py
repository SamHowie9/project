import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


# stores an empty list to contain all the image data to train the model
all_images = []

# loop through the directory containing all the image files
for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):

    # open the fits file and get the image data (this is a numpy array of each pixel value)
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

    # append the image data to the main list containing all data of all the images
    all_images.append(image)


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
encoded = Dense(units=6, name="encoded")(x)                                                         # (2)


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

autoencoder.load_weights("Weights/7_feature_weights")


