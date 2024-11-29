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



# Form reconstructions

# number of images to reconstruct
n = 12

# create a subset of the validation data to reconstruct (first 10 images)
images_to_reconstruct = test_images[n:]
# images_to_reconstruct = train_images[n:]


# create figure to hold subplots
fig, axs = plt.subplots(2, n-1, figsize=(18,5))

# plot each subplot
for i in range(0, n-1):

    # transform the image into the pca feature space
    temp_features = pca.transform(images_to_reconstruct[i])

    # transform back to form a reconstruction
    reconstructed_image = pca.inverse_transform(test_features[i])

    print(reconstructed_image.shape)

    # normalise the original image and reconstruction (for display purposes)
    original_image = normalise_independently(images_to_reconstruct)
    reconstructed_image = normalise_independently(reconstructed_image)

    # show the original image (remove axes)
    # axs[0,i].imshow(images_to_reconstruct[i])
    axs[0,i].imshow(original_image)
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    # show the reconstructed image (remove axes)
    # axs[1,i].imshow(reconstructed_images[i])
    axs[1,i].imshow(reconstructed_image)
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)

# plt.savefig("Variational Eagle/Reconstructions/Validation/normalised_independently_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_reconstruction_3")
plt.savefig("Variational Eagle/Reconstructions/Validation/normalised_independently_pca_reconstruction")
plt.show()
