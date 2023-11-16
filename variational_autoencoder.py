import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
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



# Instantiate a Keras tensor to allow us to build the model
input_image = keras.Input(shape=(256, 256, 3))

# Instantiate a Keras tensor to allow us to build the model
input_image = keras.Input(shape=(256, 256, 3))

# layers for the encoder
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 32)
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 64, 64)
x = Flatten()(x)                                                                                    # (262144) = (64 * 64 * 64)
encoded = Dense(units=2, activation="relu", name="embedded")(x)                                     # (2)

# layers for the decoder
x = Dense(units=64*64*32, activation="relu")(encoded)                                               # (131072) = (64 * 64 * 32)
x = Reshape((64, 64, 32))(x)                                                                        # (64, 64, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 64)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (265, 256, 32)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="relu", padding="same")(x)           # (256, 256, 3)



# create and compile the autoencoder model
autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=50, batch_size=1, validation_data=(test_images, test_images))