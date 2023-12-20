import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random



# set the encoding dimension (number of extracted features)
encoding_dim = 32

# set the number of clusters
n_clusters = 20

# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

# A = [1, 2, 4, 8, 16]
#
# for i in A:
#
#     # perform hierarchical ward clustering
#     hierarchical = AgglomerativeClustering(n_clusters=i, metric="euclidean", linkage="ward")
#
#     # get hierarchical clusters
#     clusters = hierarchical.fit_predict(extracted_features)
#
#     count = pd.Series(clusters).value_counts()
#     print(count.tolist())


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=None, distance_threshold=0, affinity="euclidean", linkage="ward")
hierarchical = hierarchical.fit(extracted_features)

plt.figure(figsize=(15,15))

plot_dendrogram(hierarchical, truncate_mode="lastp", p=4)
# plot_dendrogram(hierarchical, truncate_mode="level", p=4, color_threshold=0, link_color_func=lambda k:"black")

# plt.axhline(y=375)
# plt.axhline(y=235)
# plt.axhline(y=134)
# plt.axhline(y=93)

plt.savefig("Plots/hierarcial_clustering_dendrogram")
plt.show()

