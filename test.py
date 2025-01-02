import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
# import cv2
import keras
from keras import ops
from tensorflow.keras import backend as K



# A = [[1, 2], [3, 4], [5, 6], [7, 8]]
# B = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# B = [[3, 3], [5, 5], [9, 9], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4], [4, 4]]

A = np.array([10, 12, 414, 62, 62, 8, 9])
B = np.array([1, 2, 3, 4, 5, 6, 7])

print(A - B)
print(np.square(A - B))


# chosen_galaxies = np.load("Galaxy Properties/Eagle Properties/Chosen Galaxies.npy")
#
# print(len(chosen_galaxies))
#
#
# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # # account for hte validation data and remove final 200 elements
# # structure_properties.drop(structure_properties.tail(200).index, inplace=True)
# # physical_properties.drop(physical_properties.tail(200).index, inplace=True)
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
# print(len(all_properties))
#
# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# # print(bad_fit)
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# print(len(list(all_properties["GalaxyID"])))
#
# print(len(all_properties))
#
# print(list(all_properties["GalaxyID"]))



# normalise each band individually
def normalise_independently(image):
    image = image.T
    for i in range(0, 3):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
    return image.T


data_ids = [2, 13632, 21794, 23302, 24478]
reconstruction_ids = [26264, 27474, 28851, 29818, 30903]

data = []
reconstruction = []

for galaxy in data_ids:
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    data.append(normalise_independently(image))

for galaxy in reconstruction_ids:
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy) + ".png")
    reconstruction.append(normalise_independently(image))

data_image = np.expand_dims(data[0], axis=0)
reconstruction_image = np.expand_dims(reconstruction[0], axis=0)

def root_mean_squared_error(data, reconstruction):

    data = np.reshape(np.array(data), (256, 256, 3)).T
    reconstruction = np.reshape(np.array(reconstruction), (256, 256, 3)).T

    rmse_0 = np.sqrt(np.mean(np.square(reconstruction[0] - data[0])))
    rmse_1 = np.sqrt(np.mean(np.square(reconstruction[1] - data[1])))
    rmse_2 = np.sqrt(np.mean(np.square(reconstruction[2] - data[2])))

    return np.mean([rmse_0, rmse_1, rmse_2])

    # return np.sqrt(np.mean(np.square(reconstruction[0] - data[0])))



    # print(np.max(diff))
    #
    # print(pd.DataFrame(diff))
    #
    # print(data.shape)
    # print(reconstruction.shape)




    # diff = reconstruction - data
    #
    # print(pd.DataFrame(diff))
    #
    # return np.sqrt(np.mean(np.square(reconstruction - data)))



print(data_image.shape)
print(reconstruction_image.shape)

loss = root_mean_squared_error(data_image, reconstruction_image)
# loss = keras.losses.binary_crossentropy(data_image, reconstruction_image)

print(loss)