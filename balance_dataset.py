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



# select which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"




# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# load the disk-total ratios
disk_total = pd.read_csv("Galaxy Properties/Eagle Properties/disk_to_total.csv", comment="#")

# add the disk-total ratios to the other properties
all_properties = pd.merge(all_properties, disk_total, on="GalaxyID")


# # load the non-parametric properties (restructure the dataframe to match the others)
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



spirals = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"] > 0.2])
unknown = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"] < 0.1])


print(len(all_properties))

print(len(spirals), "-", len(spirals)/len(all_properties))
print(len(unknown), "-", len(unknown)/len(all_properties))
print(len(ellipticals), "-", len(ellipticals)/len(all_properties))
print()






# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 1) |
#                           (all_properties["flag_r"] == 4) |
#                           (all_properties["flag_r"] == 5) |
#                           (all_properties["flag_r"] == 6))].index.tolist()
#
# print(bad_fit)
# print(len(bad_fit))
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # reset the index values
# all_properties = all_properties.reset_index(drop=True)
#
#
#
# # spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
# # unknown = list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")])
# # ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 4])
#
# spirals = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"] > 0.2])
# unknown = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")])
# ellipticals = list(all_properties["GalaxyID"].loc[all_properties["DiscToTotal"] < 0.1])
#
#
# print(len(all_properties))
#
# print(len(spirals), "-", len(spirals)/len(all_properties))
# print(len(unknown), "-", len(unknown)/len(all_properties))
# print(len(ellipticals), "-", len(ellipticals)/len(all_properties))
# print()














# specify how the images are to be augmented
# datagen = ImageDataGenerator(rotation_range=360, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
datagen = ImageDataGenerator(rotation_range=360, fill_mode="nearest")



# augment the elliptical images

# for galaxy in ellipticals:
#
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
#     image = image.reshape(1, 256, 256, 3)
#
#     i = 0
#     for batch in datagen.flow(image, batch_size=1, save_to_dir="/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/", save_prefix=galaxy, save_format="png"):
#         i += 1
#         if i >= 6:
#             break
# print("...")



# augment the transitional images

# for galaxy in unknown:
#
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
#     image = image.reshape(1, 256, 256, 3)
#
#     i = 0
#     for batch in datagen.flow(image, batch_size=1, save_to_dir="/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/", save_prefix=galaxy, save_format="png"):
#         i += 1
#         if i >= 8:
#             break





# check the number of augmented images

augmented_ellipticals =  os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Ellipticals All/")
augmented_transitional = os.listdir("/cosma5/data/durham/dc-howi1/project/Eagle Augmented/Transitional All/")

print()
print(len(augmented_ellipticals))
print(len(augmented_transitional))
print()

for galaxy in ellipticals:
    count = 0
    for file in augmented_ellipticals:
        if file.startswith(str(galaxy)):
            count += 1
    if count < 10:
        print(galaxy, count)

print()

for galaxy in unknown:
    count = 0
    for file in augmented_transitional:
        if file.startswith(str(galaxy)):
            count += 1
    if count < 10:
        print(galaxy, count)

print()









# fig, axs = plt.subplots(6, 6, figsize=(20, 20))
#
# for i in range(0, 6):
#     for j in range(0, 6):
#
#         index = i + (6*j)
#         print(index)
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(spiral_sample[index]) + ".png")
#
#         sersic = all_properties.loc[all_properties["GalaxyID"] == spiral_sample[index], "n_r"].values[0]
#         dt = all_properties.loc[all_properties["GalaxyID"] == spiral_sample[index], "DiscToTotal"].values[0]
#
#         axs[i][j].imshow(image)
#         # axs[i][j].set_title(str(spiral_sample[index]) + ", n=" + str(sersic))
#         axs[i][j].set_title(str(spiral_sample[index]) + ", d/t=" + str(round(dt, 3)))
#         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/sample_dt_spiral", bbox_inches='tight')
# plt.show()



# fig, axs = plt.subplots(6, 6, figsize=(15, 15))
#
# for i in range(0, 6):
#     for j in range(0, 6):
#
#         index = i + (6*j)
#         print(index)
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(unknown_sample[index]) + ".png")
#
#         sersic = all_properties.loc[all_properties["GalaxyID"] == unknown_sample[index], "n_r"].values[0]
#
#         axs[i][j].imshow(image)
#         axs[i][j].set_title(str(unknown_sample[index]) + ", n=" + str(sersic))
#         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/sample_unknown", bbox_inches='tight')
# plt.show()



# fig, axs = plt.subplots(6, 6, figsize=(15, 15))
#
# for i in range(0, 6):
#     for j in range(0, 6):
#
#         index = i + (6*j)
#         print(index)
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(elliptical_sample[index]) + ".png")
#
#         sersic = all_properties.loc[all_properties["GalaxyID"] == elliptical_sample[index], "n_r"].values[0]
#         dt = all_properties.loc[all_properties["GalaxyID"] == elliptical_sample[index], "DiscToTotal"].values[0]
#
#         axs[i][j].imshow(image)
#         # axs[i][j].set_title(str(elliptical_sample[index]) + ", n=" + str(sersic))
#         axs[i][j].set_title(str(elliptical_sample[index]) + ", d/t=" + str(round(dt, 3)))
#         axs[i][j].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
#
# plt.savefig("Variational Eagle/Plots/sample_dt_elliptical", bbox_inches='tight')
# plt.show()




# spiral = 17917747
# unknown = 17752121
# elliptical = 9526568
#
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(spiral) + ".png")
# axs[0].imshow(image)
# axs[0].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
# axs[0].set_title("Disk & Spiral", fontsize=20)
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(elliptical) + ".png")
# axs[1].imshow(image)
# axs[1].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
# axs[1].set_title("Bulge (Ellitpical)", fontsize=20)
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(unknown) + ".png")
# axs[2].imshow(image)
# axs[2].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
# axs[2].set_title("Intermediate", fontsize=20)
#
# plt.savefig("Variational Eagle/Plots/sample_three_types", bbox_inches='tight')
# plt.show()











