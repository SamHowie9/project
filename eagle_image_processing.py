import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg





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





# list to contain all galaxy images
all_images = []

# load the ids of the chosen galaxies
chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/Chosen Galaxies.npy")

# # loop through each galaxy in the supplemental file
for i, galaxy in enumerate(chosen_galaxies):

    # get the filename of each galaxy in the supplemental file
    filename = "galrand_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

    original_image = image

    smallest_non_zero = np.min(image[image > 0])
    image = np.where(image == 0.0, smallest_non_zero, image)
    # print(smallest)

    # apply log transformation to the image
    image = np.log10(image)

    log_image = image

    # normalise the image (either each band independently or to the r band)
    # image = normalise_independently(image)
    image = normalise_to_r(image)

    # add the image to the dataset
    all_images.append(image)



    print(np.min(original_image.T[0]), np.max(original_image.T[0]), "  ", np.min(log_image.T[0]), np.max(log_image.T[0]), "  ", np.min(image.T[0]), np.max(image.T[0]))
    print(np.min(original_image.T[1]), np.max(original_image.T[1]), "  ", np.min(log_image.T[1]), np.max(log_image.T[1]), "  ", np.min(image.T[1]), np.max(image.T[1]))
    print(np.min(original_image.T[2]), np.max(original_image.T[2]), "  ", np.min(log_image.T[2]), np.max(log_image.T[2]), "  ", np.min(image.T[2]), np.max(image.T[2]))
    print()

    # if not(np.min(image[0]) == np.min(image[1]) == np.min(image[2]) == 0.0 and np.max(image[0]) == np.max(image[1]) == np.max(image[2]) == 1.0):
    #     print(galaxy)
    #     print(np.min(image.T[0]), np.max(image.T[0]))
    #     print(np.min(image.T[1]), np.max(image.T[1]))
    #     print(np.min(image.T[2]), np.max(image.T[2]))


# print(np.array(all_images).T.shape)
# print(np.min(np.array(all_images).T[0]), np.max(np.array(all_images).T[0]))
# print(np.min(np.array(all_images).T[1]), np.max(np.array(all_images).T[1]))
# print(np.min(np.array(all_images).T[2]), np.max(np.array(all_images).T[2]))