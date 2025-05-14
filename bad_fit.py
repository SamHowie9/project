import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os




# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                          (all_properties["flag_r"] == 4) |
                          (all_properties["flag_r"] == 5) |
                          (all_properties["flag_r"] == 6))]["GalaxyID"].tolist()

print(bad_fit)
print(len(bad_fit))

fig, axs = plt.subplots(7, 6, figsize=(12, 12))

for i in range(0, 7):

    for j in range(0, 6):

        try:
            image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(bad_fit[i + (7 * j)]) + ".png")
            axs[i][j].imshow(image)
            axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
        except:
            print(i, j, i+(7*j))


plt.savefig("Variational Eagle/Plots/bad_fit_images", bbox_inches='tight')
plt.show()