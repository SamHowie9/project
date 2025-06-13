import pandas as pd
import numpy as np
import os


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


print(all_properties)

galaxies = all_properties["GalaxyID"].values
# np.save("Galaxy Properties/Eagle Properties/galaxy_names.npy", galaxies)

all_properties.to_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv", index=False)




# pie chart





augmented_galaxies = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
galaxy_names = [galaxy.split("_")[0] for galaxy in augmented_galaxies]
print(galaxy_names)
