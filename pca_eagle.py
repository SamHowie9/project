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
    image = normalise_independently(image)
    # image = normalise_to_r(image)

    # add the image to the dataset
    all_images.append(image)






# split the data into training and testing data (200 images used for testing)
train_images = np.array(all_images[:-200])
test_images = np.array(all_images[-200:])


print(train_images.shape)

# flatten the images by reshaping the array from (3424, 256, 256, 3) to (3434, 256 * 256 * 3) = (3424, 19666608)
flattened_images = train_images.reshape(3424, 196608)



pca = PCA(n_components=0.90).fit(flattened_images)

extracted_features = pca.transform(flattened_images)

np.save("Variational Eagle/Extracted Features/PCA/pca_features_0.95", extracted_features)

plt.plot(range(1, extracted_features.shape[1] + 1), pca.explained_variance_ratio_)

plt.savefig("Plots/pca_scree")
plt.show()
