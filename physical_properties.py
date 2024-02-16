import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random
import textwrap
import math




pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)




# set the encoding dimension (number of extracted features)
encoding_dim = 44

# set the number of clusters
n_clusters = 2


# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(extracted_features)


# get hierarchical centers
clf = NearestCentroid()
clf.fit(extracted_features, clusters)
centers = clf.centroids_



physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv")
physical_properties.drop(physical_properties.tail(200).index, inplace=True)
physical_properties["Cluster"] = clusters


physical_properties["Log_Stellar_Mass"] = np.log10(physical_properties["MassType_Star"])
physical_properties["Stellar_Mass/DM_Mass"] = physical_properties["MassType_Star"]/physical_properties["MassType_DM"]


print(physical_properties)



# sns.scatterplot(data=physical_properties, x="Log_Stellar_Mass", y="Stellar_Mass/DM_Mass")
# plt.ylim(-0.1, 3)

sns.histplot(data=physical_properties, x="Stellar_Mass/DM_Mass")
plt.xlim(-0.1, 0.5)

# plt.scatter(physical_properties["MassType_Star"], physical_properties["MassType_Star"])

plt.show()
