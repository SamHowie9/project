from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
import pandas as pd
import numpy as np

# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print("Bad Fit Indices:", bad_fit)

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# get a list of all the ids of the galaxies
chosen_galaxies = list(all_properties["GalaxyID"])

random.seed(1)
random_galaxies = random.sample(chosen_galaxies, 25)



fig, axs = plt.subplots(5, 5, figsize=(15, 15))

n = 0

for i in range(0, 5):
    for j in range(0, 5):

        filename = "galrand_" + str(random_galaxies[n]) + ".png"
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)

        axs[i][j].imshow(image)
        axs[i][j].get_xaxis().set_visible(False)
        axs[i][j].get_yaxis().set_visible(False)

        n += 1

plt.savefig("Variational Eagle/Plots/random_galaxy_sample", bbox_inches='tight')
plt.show()