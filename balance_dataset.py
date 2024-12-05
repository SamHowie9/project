import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg



# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# # account for hte validation data and remove final 200 elements
# structure_properties.drop(structure_properties.tail(200).index, inplace=True)
# physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")


chosen_galaxies = list(all_properties["GalaxyID"])
print(len(chosen_galaxies))


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print(bad_fit)

# remove those galaxies
for i, galaxy in enumerate(bad_fit):
    all_properties = all_properties.drop(galaxy, axis=0)


chosen_galaxies = list(all_properties["GalaxyID"])
print(len(chosen_galaxies))

np.save("Galaxy Properties/Eagle Properties/Chosen Galaxies.npy", chosen_galaxies)


spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] > 2.5])

print(len(spirals))
print(len(ellipticals))