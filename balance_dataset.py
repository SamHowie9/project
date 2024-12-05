import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random



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


print(len(all_properties))


spirals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] <= 2.5])
unknown = list(all_properties["GalaxyID"].loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")])
ellipticals = list(all_properties["GalaxyID"].loc[all_properties["n_r"] >= 4])

print(len(spirals))
print(len(unknown))
print(len(ellipticals))


# randomly sample half the spirals (seed for reproducibility)
random.seed(1)
spirals = random.sample(spirals, round(len(spirals)/2))

fig, axs = plt.subplots(4, 4)

all_images = []

# open and add the spiral galaxies to the dataset
for galaxy in spirals:
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    all_images.append(normalise_independently(image))

# open and add the 'unknown' galaxies to the dataset (sersic index between 2.5 and 4)
for galaxy in unknown:
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    all_images.append(normalise_independently(image))

# open and add all the elliptical galaxies to the dataset
for galaxy in ellipticals:

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    all_images.append(normalise_independently(image))

    # add three more variants of this image

    # rotate the image by 90 degrees and add random noise (seed for reproducibility)
    image_rot_90 = np.rot90(image)
    random.seed(2)
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image_rot_90[0]), len(image_rot_90[0])))
        image_rot_90.T[i] = image_rot_90.T[i] + gaussian

    # rotate the original image by 180 degrees (90 twice) and add different random noise (seed for reproducibility)
    image_rot_180 = np.rot90(np.rot90(image))
    random.seed(3)
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image_rot_180[0]), len(image_rot_180[0])))
        image_rot_180.T[i] = image_rot_180.T[i] + gaussian

    # rotate the original image by 270 degrees (90 thrice) and add different random noise (seed for reproducibility)
    image_rot_270 = np.rot90(np.rot90(np.rot90(image)))
    random.seed(4)
    for i in range(0, 3):
        gaussian = np.random.normal(0, 0.1, (len(image_rot_270[0]), len(image_rot_270[0])))
        image_rot_270.T[i] = image_rot_270.T[i] + gaussian

    # add the three variants to the dataset
    all_images.append(normalise_independently(image_rot_90))
    all_images.append(normalise_independently(image_rot_180))
    all_images.append(normalise_independently(image_rot_270))


    if galaxy == 0 or galaxy == 1 or galaxy == 2 or galaxy == 3:

        axs[galaxy, 0].imshow(normalise_independently(image))
        axs[galaxy, 0].get_xaxis().set_visible(False)
        axs[galaxy, 0].get_yaxis().set_visible(False)

        axs[galaxy, 1].imshow(normalise_independently(image_rot_90))
        axs[galaxy, 1].get_xaxis().set_visible(False)
        axs[galaxy, 1].get_yaxis().set_visible(False)

        axs[galaxy, 2].imshow(normalise_independently(image_rot_180))
        axs[galaxy, 2].get_xaxis().set_visible(False)
        axs[galaxy, 2].get_yaxis().set_visible(False)

        axs[galaxy, 3].imshow(normalise_independently(image_rot_270))
        axs[galaxy, 3].get_xaxis().set_visible(False)
        axs[galaxy, 3].get_yaxis().set_visible(False)



# randomly (seed for reproducibility) rearrange the galaxies
random.seed(5)
random.shuffle(all_images)


plt.savefig("Variational Eagle/Plots/elliptical_variations")
plt.show()