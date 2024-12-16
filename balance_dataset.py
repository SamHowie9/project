import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
import os



# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T



# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")




# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
print(bad_fit)

# remove those galaxies
for i, galaxy in enumerate(bad_fit):
    all_properties = all_properties.drop(galaxy, axis=0)


spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
unknown = list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 4])


print(len(spirals))
print(len(unknown))
print(len(ellipticals))

print(ellipticals)

ellipticals = [9768388]

datagen = ImageDataGenerator(rotation_range=360, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")


for galaxy in ellipticals:

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir="/cosma7/data/durham/dc-howi1/project/Eagle Augmented/", save_prefix=galaxy, save_format="png"):
        i += 1
        if i > 29:
            break




galaxy = 9768388

files = os.listdir("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/")

print(files)

fig, axs = plt.subplots(4, 10)

image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")

axs[0, 0].imshow(image)

for i in range(0, 10):

    image_1 = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/" + files[i])
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.01, (len(image_1[0]), len(image_1[0])))
        image_1.T[i] = image_1.T[i] + gaussian
    image_1 = normalise_independently(image_1)


    image_2 = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/" + files[i + 10])
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.01, (len(image_2[0]), len(image_2[0])))
        image_2.T[i] = image_2.T[i] + gaussian
    image_2 = normalise_independently(image_2)


    image_3 = mpimg.imread("/cosma7/data/durham/dc-howi1/project/Eagle Augmented/" + files[i + 20])
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.01, (len(image_3[0]), len(image_3[0])))
        image_3.T[i] = image_3.T[i] + gaussian
    image_3 = normalise_independently(image_3)


    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)

    axs[1, i].imshow(image_1)
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].get_yaxis().set_visible(False)

    axs[2, i].imshow(image_2)
    axs[2, i].get_xaxis().set_visible(False)
    axs[2, i].get_yaxis().set_visible(False)

    axs[3, i].imshow(image_3)
    axs[3, i].get_xaxis().set_visible(False)
    axs[3, i].get_yaxis().set_visible(False)

plt.savefig("Variational Eagle/Plots/balancing")
plt.show()






# # randomly sample half the spirals (seed for reproducibility)
# random.seed(1)
# spirals = random.sample(spirals, round(len(spirals)/2))
#
# fig, axs = plt.subplots(4, 4)
#
# all_images = []
#
# # open and add the spiral galaxies to the dataset
# for galaxy in spirals:
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
#     all_images.append(normalise_independently(image))
#
# # open and add the 'unknown' galaxies to the dataset (sersic index between 2.5 and 4)
# for galaxy in unknown:
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
#     all_images.append(normalise_independently(image))
#
# # open and add all the elliptical galaxies to the dataset
# for n, galaxy in enumerate(ellipticals):
#
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
#     all_images.append(normalise_independently(image))
#
#     # add three more variants of this image
#
#     # rotate the image by 90 degrees and add random noise
#     image_rot_90 = np.rot90(np.copy(image), k=1)
#     for i in range(0, 3):
#         gaussian = np.random.normal(0, 0.01, (len(image_rot_90[0]), len(image_rot_90[0])))
#         image_rot_90.T[i] = image_rot_90.T[i] + gaussian
#
#     # flip the original image horizontally and add random noise
#     image_flip = np.fliplr(np.copy(image))
#     for i in range(0, 3):
#         gaussian = np.random.normal(0, 0.01, (len(image_flip[0]), len(image_flip[0])))
#         image_flip.T[i] = image_flip.T[i] + gaussian
#
#     # flip the rotated image horizontally and add random noise
#     image_flip_90 = np.fliplr(np.copy(image_rot_90))
#     for i in range(0, 3):
#         gaussian = np.random.normal(0, 0.01, (len(image_flip_90[0]), len(image_flip_90[0])))
#         image_flip_90.T[i] = image_flip_90.T[i] + gaussian
#
#     # add the three variants to the dataset
#     all_images.append(normalise_independently(image_rot_90))
#     all_images.append(normalise_independently(image_flip))
#     all_images.append(normalise_independently(image_flip_90))
#
#
#
# chosen = [1852, 1856, 1860, 1864]
#
# for i, galaxy in enumerate(chosen):
#
#     axs[i, 0].imshow(all_images[galaxy])
#     axs[i, 1].imshow(all_images[galaxy + 1])
#     axs[i, 2].imshow(all_images[galaxy + 2])
#     axs[i, 3].imshow(all_images[galaxy + 3])
#
#
#
# # randomly (seed for reproducibility) rearrange the galaxies
# # random.seed(5)
# # random.shuffle(all_images)
#
# # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(spirals[0]) + ".png")
# # axs[0, 0].imshow(image)
# # axs[1, 0].imshow(all_images[0])
#
# plt.savefig("Variational Eagle/Plots/elliptical_variations_not_normalised")
# plt.show()