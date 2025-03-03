import zipfile
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
import random
import os


# with zipfile.ZipFile("gz2_images_all.zip", "r") as zip_ref:
#     zip_ref.extractall("Galaxy Zoo Images")



# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T


# load the original dataset

# list to store all the images
all_images = []

# list of the names of all galaxy images
galaxies = os.listdir("/cosma7/data/durham/dc-howi1/project/project/Galaxy Zoo Images/gz2_images_all/")

# # loop through each galaxy in the supplemental file
for i, galaxy in enumerate(galaxies):

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/durham/dc-howi1/project/project/Galaxy Zoo Images/gz2_images_all/" + galaxy).astype('float64')

    # shrink the image to 256x256 using area interpolation (best for shrinking images)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # normalise each band independently between 0 and 1
    image = normalise_independently(image)

    # add the image to the dataset
    all_images.append(image)

all_images = np.array(all_images)
print(all_images.shape)

random.seed(1)
image_sample_index = random.sample(range(0, len(all_images)-1), 25)
image_sample = [all_images[i] for i in image_sample_index]


fig, axs = plt.subplots(5, 5, figsize=(10,10))

n = 0

for i in range(0, 5):
    for j in range(0, 5):

        axs[i, j].imshow(image_sample[n], vmin=0, vmax=1)
        axs[i, j].get_xaxis().set_visible(False)
        axs[i, j].get_yaxis().set_visible(False)

        n += 1

plt.savefig("Variational Zoo/Plots/galaxy_zoo_sample_2")
plt.show()

