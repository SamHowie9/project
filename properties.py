import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
#
# print(augmented_galaxies[0])
# print(augmented_galaxies[0].split("_"))
# print(augmented_galaxies[0].split("_")[0])
#
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# np.save("Galaxy Properties/Eagle Properties/augmented_elliptical_all", galaxy_names)
#
# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# np.save("Galaxy Properties/Eagle Properties/augmented_transitional_all", galaxy_names)






# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# load the non parametric properties (restructure the dataframe to match the others)
non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
non_parametric_properties = non_parametric_properties.reset_index()
non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")

# load the disk-total ratios
disk_total = pd.read_csv("Galaxy Properties/Eagle Properties/disk_to_total.csv", comment="#")

# add the non parametric properties, and the disk-total to the other properties dataframe
all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")
all_properties = pd.merge(all_properties, disk_total, on="GalaxyID")

print(all_properties.shape)

# drop invalid d/t ratio
all_properties = all_properties[all_properties["DiscToTotal"] >= 0]

print(all_properties.shape)

all_properties.to_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")

galaxy_names = all_properties["GalaxyID"].tolist()
np.save("Galaxy Properties/Eagle Properties/chosen_glaxies.npy", galaxy_names)

# print(all_properties)
#
# galaxies = all_properties["GalaxyID"].values
# # np.save("Galaxy Properties/Eagle Properties/galaxy_names.npy", galaxies)
#
# # all_properties.to_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv", index=False)
#
#
# # for i in range(2000, len(galaxies)):
# #     print(galaxies[i])
#
# elliptical_names = np.load("Galaxy Properties/Eagle Properties/augmented_elliptical_all.npy")
# transitional_names = np.load("Galaxy Properties/Eagle Properties/augmented_transitional_all.npy")
#
# print(len(elliptical_names))
#
# elliptical_set = set(elliptical_names)
# print(len(elliptical_set))
# print(elliptical_set)
#
# # for unique in elliptical_set:
# #
# #     count = 0
# #
# #     for galaxy in elliptical_names:
# #         if unique == galaxy:
# #             count += 1
# #
# #     if count != 6:
# #         print(unique)
#
# # print(elliptical_names)
#
#
# all_properties_balanced = all_properties.copy()
#
# # print(all_properties_all)
#
# for galaxy in elliptical_names:
#
#     properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
#     all_properties_balanced.loc[len(all_properties_balanced)] = properties
#
# print(all_properties_balanced)
#
# for galaxy in transitional_names:
#
#     properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
#     all_properties_balanced.loc[len(all_properties_balanced)] = properties
#
# print(all_properties_balanced)
#
#
# all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")
#
# # pie chart






