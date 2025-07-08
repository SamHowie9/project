import pandas as pd
import numpy as np
import tables
import os

# print(np.__version__)
# print(tables.__version__)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_elliptical_all", galaxy_names)
#
# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_transitional_all", galaxy_names)
#
#
#
# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Spirals Only/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_spiral_only", galaxy_names)
#
# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals Only/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_elliptical_only", galaxy_names)
#
# augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional Only/")
# galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
# # print(galaxy_names)
# print(len(galaxy_names))
# np.save("Galaxy Properties/Eagle Properties/augmented_transitional_only", galaxy_names)



print()





# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# print(structure_properties)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

print(all_properties.shape)

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
print(all_properties)

# print(all_properties)

all_properties.to_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv", index=False)

galaxy_names = all_properties["GalaxyID"].tolist()
np.save("Galaxy Properties/Eagle Properties/chosen_glaxies.npy", galaxy_names)

print()








all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")

elliptical_names = np.load("Galaxy Properties/Eagle Properties/augmented_elliptical_all.npy")
transitional_names = np.load("Galaxy Properties/Eagle Properties/augmented_transitional_all.npy")

all_properties_balanced = all_properties.copy()

print(all_properties_balanced.shape)

# print(all_properties_balanced)

print(elliptical_names)

for galaxy in elliptical_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    all_properties_balanced.loc[len(all_properties_balanced)] = properties

print(all_properties_balanced.shape)

for galaxy in transitional_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    all_properties_balanced.loc[len(all_properties_balanced)] = properties

print(all_properties_balanced.shape)


all_properties_balanced.to_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv", index=False)

print()









all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")

spiral_names = np.load("Galaxy Properties/Eagle Properties/augmented_spiral_only.npy")

all_properties_spirals = all_properties.copy()

print(all_properties_spirals.shape)
all_properties_spirals = all_properties_spirals[all_properties_spirals["DiscToTotal"] > 0.2].reset_index(drop=True)
# all_properties_spirals = all_properties_spirals.reset_index()
print(all_properties_spirals.shape)

for galaxy in spiral_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    # print(len(properties))
    all_properties_spirals.loc[len(all_properties_spirals)] = properties

print(all_properties_spirals.shape)

all_properties_spirals.to_csv("Galaxy Properties/Eagle Properties/all_properties_spirals.csv", index=False)

print()








all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")

elliptical_names = np.load("Galaxy Properties/Eagle Properties/augmented_elliptical_only.npy")

all_properties_ellipticals = all_properties.copy()

print(all_properties_ellipticals.shape)
all_properties_ellipticals = all_properties_ellipticals[all_properties_ellipticals["DiscToTotal"] < 0.1].reset_index(drop=True)
print(all_properties_ellipticals.shape)

for galaxy in elliptical_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    # print(properties)
    all_properties_ellipticals.loc[len(all_properties_ellipticals)] = properties

print(all_properties_ellipticals.shape)

all_properties_ellipticals.to_csv("Galaxy Properties/Eagle Properties/all_properties_ellipticals.csv", index=False)

print()









all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")

transitional_names = np.load("Galaxy Properties/Eagle Properties/augmented_transitional_only.npy")

all_properties_transitional = all_properties.copy()

print(all_properties_transitional.shape)
all_properties_transitional = all_properties_transitional[all_properties_transitional["DiscToTotal"].between(0.1, 0.2, inclusive="both")].reset_index(drop=True)
print(all_properties_transitional.shape)

for galaxy in transitional_names:

    properties = all_properties[all_properties["GalaxyID"] == int(galaxy)].iloc[0].tolist()
    # print(properties)
    all_properties_transitional.loc[len(all_properties_transitional)] = properties

print(all_properties_transitional.shape)

all_properties_transitional.to_csv("Galaxy Properties/Eagle Properties/all_properties_transitional.csv", index=False)

print()






