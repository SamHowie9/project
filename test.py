import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                          (all_properties["flag_r"] == 4) |
                          (all_properties["flag_r"] == 5) |
                          (all_properties["flag_r"] == 6))].index.tolist()

print("Bad Fit Indices:", bad_fit)

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# get a list of all the ids of the galaxies
chosen_galaxies = list(all_properties["GalaxyID"])


spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
unknown = list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 4])

print("Original All", len(all_properties))

print("Spirals", len(spirals), "-", len(spirals)/len(all_properties))
print("Unknown", len(unknown), "-", len(unknown)/len(all_properties))
print("Ellipticals", len(ellipticals), "-", len(ellipticals)/len(all_properties))
print()





# list to contain all galaxy images
all_images = []

# # loop through each galaxy
for i, galaxy in enumerate(chosen_galaxies):
    # get the filename of each galaxy in the supplemental file
    filename = "galrand_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

    # normalise the image (each band independently)
    image = normalise_independently(image)

    # add the image to the dataset
    all_images.append(image)

print("Original Images", len(all_images))

# split the data into training and testing data (200 images used for testing)
train_images = all_images[:-200]
test_images = np.array(all_images[-200:])

print("Training Set", len(train_images))
print("Testing Set", len(test_images))
print()

# load the filenames of the augmented elliptical images
augmented_galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/")

print("Augmented Ellipticals", len(augmented_galaxies))

for galaxy in augmented_galaxies:
    # load each augmented image
    image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)

print("Training Set", len(train_images))
print()

# load the filenames of the augmented unknown images
augmented_galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/")


print("Augmented Unknown", len(augmented_galaxies))

for galaxy in augmented_galaxies:
    # load each augmented image
    image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/" + galaxy)

    # normalise the image
    image = normalise_independently(image)

    # add the image to the training set (not the testing set)
    train_images.append(image)

# convert the training set to a numpy array
train_images = np.array(train_images)

print("Training Set", train_images.shape)
print("Testing Set", test_images.shape)
print()
