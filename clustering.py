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
encoding_dim = 28

# set the number of clusters
n_clusters = 4

# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features_3.npy")





extracted_features_switch = extracted_features.T


# chose which features to use for clustering
meaningful_features = [8, 11, 12, 13, 14, 15, 16, 18, 20, 21]  # 24

chosen_features = []

for feature in meaningful_features:
    chosen_features.append(list(extracted_features_switch[feature]))

chosen_features = np.array(chosen_features).T


# chosen_features = extracted_features


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(chosen_features)





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


plt.figure(figsize=(20,15))

# plot_dendrogram(hierarchical, truncate_mode="lastp", p=464 )
plot_dendrogram(hierarchical, truncate_mode="level", p=5, color_threshold=0, link_color_func=lambda k:"black")

plt.ylabel("Dissimilarity", fontsize=25, labelpad=10)
plt.xlabel("Number of Images in Clusters", fontsize=25, labelpad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)

# plt.axhline(y=95, label="Cutoff Points")
# plt.axhline(y=120)
# plt.axhline(y=135)
# plt.axhline(y=180)
# plt.axhline(y=250)
# plt.axhline(y=26)

# plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', fontsize=20)

# plt.axhline(y=375)
# plt.axhline(y=235)
# plt.axhline(y=134)
# plt.axhline(y=93)

plt.savefig("Plots/hierarchical_clustering_dendrogram_select_features")
plt.show()

