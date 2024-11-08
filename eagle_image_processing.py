import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg



# stores an empty list to contain all the image data to train the model
all_images = []

# load the supplemental file into a dataframe
df = pd.read_csv("Galaxy Properties/Eagle Properties/stab3510_supplemental_file/table1.csv", comment="#")

# loop through each galaxy in the supplemental file
for i, galaxy in enumerate(df["GalaxyID"].tolist()):

    # get the filename of each galaxy in the supplemental file
    filename = "galrand_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

    smallest_non_zero = np.min(image[image > 0])
    image = np.where(image == 0.0, smallest_non_zero, image)
    # print(smallest)

    # apply log transformation to the image
    image = np.log10(image)

    # normalise the image (either each band independently or to the r band)
    image = normalise_independently(image)
    # image = normalise_to_r(image)

    # add the image to the dataset
    all_images.append(image)



    print(np.min(image.T[0]), np.max(image.T[0]))
    print(np.min(image.T[1]), np.max(image.T[1]))
    print(np.min(image.T[2]), np.max(image.T[2]))
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