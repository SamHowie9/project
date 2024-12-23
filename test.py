import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
# import cv2


A = [[1, 2], [3, 4], [5, 6], [7, 8]]
B = [1, 2, 3, 4, 5, 6, 7, 8, 9]

B = [[3, 3], [5, 5], [9, 9], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4], [4, 4]]



chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/Chosen Galaxies.npy")

print(len(chosen_galaxies))


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# # account for hte validation data and remove final 200 elements
# structure_properties.drop(structure_properties.tail(200).index, inplace=True)
# physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

print(len(all_properties))

# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# print(bad_fit)

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

print(len(list(all_properties["GalaxyID"])))

print(len(all_properties))