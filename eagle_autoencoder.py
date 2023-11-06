import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
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
        image = fits.open("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)

        # append the image data to the main list containing all data of all the images
        train_images.append(image)

    # return this list
    return train_images


# get the images and labels to train the model
train_images = get_images()

[[[][][]]]

# find the number of images that you will test the model on
testing_count = int(len(train_images/4))

# split the data into training and testing data based on this number (and convert from list to numpy array of shape (256,256,3) given it is an rgb image
train_images = np.array(train_images[:testing_count*3])
test_images = np.array(train_images[testing_count:])


# Instantiate a Keras tensor to allow us to build the model
input_image = keras.Input(shape=(256, 256, 3))

# layers for the encoder
encoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(input_image)

# layers for the decoder (extra one with 1 filter to get back to the correct shape)
decoded = Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(encoded)
decoded = Conv2D(filters=3, kernel_size=3, activation="sigmoid", padding="same")(encoded)


# create and compile the autoencoder model
autoencoder = keras.Model(input_image, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# train the model
model_data = autoencoder.fit(train_images, train_images, epochs=50, batch_size=1, validation_data=(test_images, test_images))



# create a subset of the validation data to reconstruct (first 10 images)
images_to_reconstruct = test_images[:10]

# number of images to reconstruct
n = 10

# reconstruct the images
reconstructed_images = autoencoder.predict(test_images[:n])

# create figure to hold subplots
plt.figure(figsize=(20,4))

# plot each subplot
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
plt.savefig("Plots/eagle_reconstruction")