from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
from keras import backend as K
import numpy as np
# import IPython
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)




# set the encoding dimension (number of extracted features)
encoding_dim = 28

# set the number of clusters
n_clusters = 10



# Define keras tensor for the encoder
input_image = keras.Input(shape=(128, 128, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=64)(x)                                                                              # (32)
encoded = Dense(units=encoding_dim, name="encoded")(x)                                              # (2)


# layers for the decoder
x = Dense(units=64)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
# x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)


# crate autoencoder
autoencoder = keras.Model(input_image, decoded)
autoencoder.summary()



# create the encoder using the autoencoder layers
encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])
encoder.summary()

print()

# create the decoder using the autoencoder layers
decoder = keras.Sequential()
for i in range(9, 17):
    decoder.add(autoencoder.layers[i])

print()

# build the decoder
decoder.build(input_shape=(None, encoding_dim))
decoder.summary()



# root means squared loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss=root_mean_squared_error)

# load the weights
autoencoder.load_weights("Weights Rand/" + str(encoding_dim) + "_feature_weights_3.h5")





# load the extracted features
extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_3.npy")
extracted_features_switch = np.flipud(np.rot90(extracted_features))






extracted_features_switch = extracted_features.T


# chose which features to use for clustering
meaningful_features = [8, 11, 12, 13, 14, 15, 16, 18, 20, 21]  # 24

chosen_features = []

for feature in meaningful_features:
    chosen_features.append(list(extracted_features_switch[feature]))

chosen_features = np.array(chosen_features).T


# chosen_features = extracted_features


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(chosen_features)

# get hierarchical centers
clf = NearestCentroid()
clf.fit(chosen_features, clusters)
centers = clf.centroids_




# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

all_properties["Cluster"] = clusters




med_extracted_feature = pd.DataFrame()


print(extracted_features[0].shape)

for i in range(0, n_clusters):

    cluster_indices = all_properties.index[all_properties["Cluster"] == i].tolist()

    med_features = []

     for feature in range(encoding_dim):

         for index in cluster_indices:

             feature = extracted_features[index][feature]



