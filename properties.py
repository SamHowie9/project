import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_elliptical_all", galaxy_names)

# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_transitional_all", galaxy_names)



augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Spirals Only/")
galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# print(galaxy_names)
print(len(galaxy_names))
np.save("Galaxy Properties/Eagle Properties/augmented_spiral_only", galaxy_names)

augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals Only/")
galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# print(galaxy_names)
print(len(galaxy_names))
np.save("Galaxy Properties/Eagle Properties/augmented_elliptical_only", galaxy_names)

augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional Only/")
galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# print(galaxy_names)
print(len(galaxy_names))
np.save("Galaxy Properties/Eagle Properties/augmented_transitional_only", galaxy_names)







# load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
# # load the non parametric properties (restructure the dataframe to match the others)
# non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
# non_parametric_properties = non_parametric_properties.reset_index()
# non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")
#
# # load the disk-total ratios
# disk_total = pd.read_csv("Galaxy Properties/Eagle Properties/disk_to_total.csv", comment="#")
#
# # add the non parametric properties, and the disk-total to the other properties dataframe
# all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")
# all_properties = pd.merge(all_properties, disk_total, on="GalaxyID")
#
# print(all_properties.shape)
#
# # drop invalid d/t ratio
# all_properties = all_properties[all_properties["DiscToTotal"] >= 0]
#
# print(all_properties.shape)
#
# all_properties.to_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")
#
# galaxy_names = all_properties["GalaxyID"].tolist()
# np.save("Galaxy Properties/Eagle Properties/chosen_glaxies.npy", galaxy_names)









# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")
#
# elliptical_names = np.load("Galaxy Properties/Eagle Properties/augmented_elliptical_all.npy")
# transitional_names = np.load("Galaxy Properties/Eagle Properties/augmented_transitional_all.npy")
#
# all_properties_balanced = all_properties.copy()
#
# print(all_properties_balanced.shape)
#
# for galaxy in elliptical_names:
#
#     properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
#     all_properties_balanced.loc[len(all_properties_balanced)] = properties
#
# print(all_properties_balanced.shape)
#
# for galaxy in transitional_names:
#
#     properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
#     all_properties_balanced.loc[len(all_properties_balanced)] = properties
#
# print(all_properties_balanced.shape)
#
#
# all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")











all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

spiral_names = np.load("Galaxy Properties/Eagle Properties/augmented_spiral_only.npy")

all_properties_balanced = all_properties.copy()

all_properties_balanced = all_properties_balanced[all_properties_balanced["DiscToTotal"] > 0.2]

print(all_properties_balanced.shape)

for galaxy in spiral_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    all_properties_balanced.loc[len(all_properties_balanced)] = properties

print(all_properties_balanced.shape)

all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_spirals.csv")









all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

elliptical_names = np.load("Galaxy Properties/Eagle Properties/augmented_elliptical_only.npy")

all_properties_balanced = all_properties.copy()

all_properties_balanced = all_properties_balanced[all_properties_balanced["DiscToTotal"] < 0.2]

print(all_properties_balanced.shape)

for galaxy in elliptical_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    all_properties_balanced.loc[len(all_properties_balanced)] = properties

print(all_properties_balanced.shape)

all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_ellipticals.csv")











all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

transitional_names = np.load("Galaxy Properties/Eagle Properties/augmented_transitional_only.npy")

all_properties_balanced = all_properties.copy()

all_properties_balanced = all_properties_balanced[all_properties_balanced["DiscToTotal"].between(0.1, 0.2, inclusive="both")]

print(all_properties_balanced.shape)

for galaxy in transitional_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    all_properties_balanced.loc[len(all_properties_balanced)] = properties

print(all_properties_balanced.shape)


all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_transitional.csv")








