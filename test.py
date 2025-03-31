import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf


encoding_dim = 24
run = 2
epochs = 750
batch_size = 32

# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")


# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

# load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_" + str(run) + ".npy")[0]
extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
encoding_dim = extracted_features.shape[1]
extracted_features_switch = extracted_features.T

# print(extracted_features.shape)

extracted_features = extracted_features[:len(all_properties)]


print(extracted_features.shape)
disk_structures = np.array(all_properties["n_r"] <= 2.5)
extracted_features = extracted_features[disk_structures]
print(extracted_features.shape)