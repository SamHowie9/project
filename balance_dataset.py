import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
import os





pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)





# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# # load the non parametric properties (restructure the dataframe to match the others)
# non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
# non_parametric_properties = non_parametric_properties.reset_index()
# non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")
#
# # add the non parametric properties to the other properties dataframe
# all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")





# print(all_properties[all_properties["flag_r"] == 5])
# print()

# print(all_properties)
#
# print(len(all_properties[all_properties["flag"] == 1]))
# print(len(all_properties[all_properties["flag_sersic"] == 1]))
#
# print(len(all_properties[(all_properties["flag"] == 1) & (all_properties["flag_sersic"] == 1)]))


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# reset the index values
all_properties = all_properties.reset_index(drop=True)


# print(all_properties["flag"].unique())
# print(all_properties["flag_sersic"].unique())
#
# print(len(all_properties[all_properties["flag"] == 1]))
# print(len(all_properties[all_properties["flag_sersic"] == 1]))







spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
unknown = list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 4])



print(len(all_properties))

print(len(spirals))
print(len(unknown))
print(len(ellipticals))



# specify how the images are to be augmented (for both unknown and ellipticals)
datagen = ImageDataGenerator(rotation_range=360, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")



# augment the elliptical images

for galaxy in ellipticals:

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    image = image.reshape(1, 256, 256, 3)

    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir="/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals/", save_prefix=galaxy, save_format="png"):
        i += 1
        if i >= 25:
            break



# augment the 'unknown' images

for galaxy in unknown:

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    image = image.reshape(1, 256, 256, 3)

    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir="/cosma7/data/durham/dc-howi1/project/Eagle Augmented/Unknown/", save_prefix=galaxy, save_format="png"):
        i += 1
        if i >= 6:
            break