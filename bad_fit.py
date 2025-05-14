import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os




# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 1) |
#                           (all_properties["flag_r"] == 4) |
#                           (all_properties["flag_r"] == 5) |
#                           (all_properties["flag_r"] == 6))]["GalaxyID"].tolist()
#
# print(bad_fit)
# print(len(bad_fit))
#
# fig, axs = plt.subplots(7, 6, figsize=(12, 12))
#
# for i in range(0, 7):
#
#     for j in range(0, 6):
#
#         try:
#             image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(bad_fit[i + (7 * j)]) + ".png")
#             axs[i][j].imshow(image)
#         except:
#             print(i, j, i+(7*j))
#
#         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
#
# plt.savefig("Variational Eagle/Plots/bad_fit_images", bbox_inches='tight')
# plt.show()








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
print()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)


# take only the sprial galaxies
all_properties = all_properties[all_properties["n_r"] <= 2.5]

# get a list of all the ids of the galaxies
chosen_galaxies = list(all_properties["GalaxyID"])

# list to contain all galaxy images
all_images = []

# # loop through each galaxy
for i, galaxy in enumerate(chosen_galaxies):

    # get the filename of each galaxy in the supplemental file
    filename = "galrand_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

    # normalise the image (each band independently)
    # image = normalise_independently(image)

    # add the image to the dataset
    all_images.append(image)

all_images = np.array(all_images)

print(np.min(all_images.T[0]), np.max(all_images.T[0]))
print(np.min(all_images.T[1]), np.max(all_images.T[1]))
print(np.min(all_images.T[2]), np.max(all_images.T[2]))

print()
print()

for image in all_images:
    print(np.min(image.T[0]), np.max(image.T[0]), "   ", np.min(image.T[1]), np.max(image.T[1]), "   ", np.min(image.T[2]), np.max(image.T[2]))