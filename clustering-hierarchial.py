import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random



# set the encoding dimension (number of extracted features)
encoding_dim = 38

# set the number of clusters
n_clusters = 4

# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features_3.npy")





extracted_features_switch = extracted_features.T


# chose which features to use for clustering
# meaningful_features = [8, 11, 12, 13, 14, 15, 16, 18, 20, 21]  # 24
meaningful_features = [1, 2, 3, 4, 7, 8, 12, 20, 24, 26, 28]  # 26

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

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



print(sns.color_palette("colorblind").as_hex())
print(sns.color_palette("pastel").as_hex())


plt.rc("text", usetex=True)

plt.figure(figsize=(20,15))

set_link_color_palette(["#80d2fe"])
plot_dendrogram(hierarchical, truncate_mode="level", p=5, color_threshold=15.5, above_threshold_color="#016398")

# set_link_color_palette(["#fcd082"])
# plot_dendrogram(hierarchical, truncate_mode="level", p=5, color_threshold=15.5, above_threshold_color="#af7004")

plt.ylabel("Dissimilarity", fontsize=25, labelpad=10)
plt.xlabel("Number of Images in Clusters", fontsize=25, labelpad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)

# plt.axhline(y=95, label="Cutoff Points")
# plt.axhline(y=15.5)

# plt.axhline(y=20.5, label="Cutoff Point")
plt.axhline(y=15.5, c="black", label="Cutoff Point")
plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', prop={"size":25})
# plt.axvline(x=160)
# plt.axvline(x=480)
plt.text(x=160, y=32, s="Less Featured", horizontalalignment='center', fontsize=25)
plt.text(x=480, y=48, s="More Featured", horizontalalignment='center', fontsize=25)

# plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', fontsize=20)

# plt.axhline(y=375)
# plt.axhline(y=235)
# plt.axhline(y=134)
# plt.axhline(y=93)

plt.savefig("Plots/hierarchical_clustering_dendrogram_less_featured", bbox_inches='tight')
plt.show()

