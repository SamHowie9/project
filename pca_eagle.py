from sklearn.decomposition import PCA
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

    # find smallest non zero pixel value in the image and replace all zero values with this (for log transformation)
    smallest_non_zero = np.min(image[image > 0])
    image = np.where(image == 0.0, smallest_non_zero, image)


    # normalise the image (either each band independently or to the r band)
    # image = normalise_independently(image)
    image = normalise_to_r(image)

    # add the image to the dataset
    all_images.append(image)






# split the data into training and testing data (200 images used for testing)
train_images = np.array(all_images[:-200])
test_images = np.array(all_images[-200:])


print(train_images.shape)



# pca = PCA(n_components=30).fit(train_images)
#
# plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)
#
# plt.show()