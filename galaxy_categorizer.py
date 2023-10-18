from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from astropy.io import fits
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

# find half the number of images (and truncate this to ensure we have an integer value but the training and testing data share no common values)
half_images = int(len(train_images)/4)

# split the images and labels in half for training and testing, convert these lists into numpy arrays
test_images = np.array(train_images[half_images:])
test_labels = np.array(train_labels[half_images:])
train_images = np.array(train_images[:half_images*3])
train_labels = np.array(train_labels[:half_images*3])

# expand the dimensions of the images from (50,50) to (50,50,1)
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)


# setting the parameters of the cnn
num_filters = 8
filter_size = 3
pool_size = 2

# Build the model using those parameters
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(50, 50, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(2, activation='softmax'),
])


# compile the model
model.compile(
    'adam',                               # gradient based optimiser
    loss='categorical_crossentropy',      # >2 classes???
    metrics=['accuracy'],                 # classification problem
)


# train the model on the images and labels, run the model 5 times
model_data = model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=30,
    batch_size=1,
    validation_data=(test_images, to_categorical(test_labels)),
)


training_accuracy = plt.plot(model_data.history["accuracy"], label="Training Data")
validation_accuracy = plt.plot(model_data.history["val_accuracy"], label="Validation Data")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()
