import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random
import textwrap


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)



# set the encoding dimension (number of extracted features)
encoding_dim = 38

# set the number of clusters
n_clusters = 11

# load the extracted features
extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_3.npy")


print(extracted_features[0].shape)


extracted_features_switch = extracted_features.T


# chose which features to use for clustering
# meaningful_features = [8, 11, 12, 13, 14, 15, 16, 18, 20, 21]   # 24
# meaningful_features = [1, 2, 7, 10, 16, 20, 23, 27, 29, 36]  # 19
meaningful_features = [1, 2, 3, 4, 7, 8, 12, 20, 24, 26, 28]  # 26

chosen_features = []

for feature in meaningful_features:
    chosen_features.append(list(extracted_features_switch[feature]))

chosen_features = np.array(chosen_features).T



# chosen_features = extracted_features


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(chosen_features)


# kmeans = KMeans(n_clusters=n_clusters)
# clusters = kmeans.fit_predict(chosen_features)

# spectral = SpectralClustering(n_clusters=n_clusters)
# clusters = spectral.fit_predict(chosen_features)

# hdbscan = HDBSCAN(metric="euclidean")
# clusters = hdbscan.fit_predict(chosen_features)


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




# Apparent Magnitude, Stellar Mass, Semi-Major Axis, Sersic Index
# columns = ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]

columns = ["n_r", "pa_r", "q_r", "re_r", "mag_r", "MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]

med_df = pd.DataFrame(columns=columns)

for i in range(0, n_clusters):
    med_cluster = all_properties.loc[all_properties["Cluster"] == i, columns].median()
    med_df.loc[i] = med_cluster

med_df["Cluster"] = list(range(0, n_clusters))

extracted_features_switch = np.flipud(np.rot90(extracted_features))
centers_switch = np.flipud(np.rot90(centers))



# print(all_properties)



order_property = "n_r"
property = "n_r"

order = med_df[order_property].sort_values(ascending=False).index.to_list()


# single property
a1 = sns.boxplot(data=all_properties, x="Cluster", y=property, showfliers=False, whis=1, palette="colorblind", order=order)
# a1 = sns.histplot(data=all_properties, x=property, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)

# plt.savefig("Plots/" + str(encoding_dim) + "_feature_3_" + str(n_clusters) + "_cluster_sersic_distribution_all_features")
plt.savefig("Plots/" + str(encoding_dim) + "_feature_3_" + str(n_clusters) + "_cluster_sersic_distribution_select_features")
plt.show()




# # structure measurements
# fig, axs = plt.subplots(1, 3, figsize=(30, 10))
#
# a1 = sns.boxplot(ax=axs[0], data=all_properties, x="Cluster", y="n_r", showfliers=False, whis=1, palette="colorblind", order=order)
# a2 = sns.boxplot(ax=axs[1], data=all_properties, x="Cluster", y="pa_r", showfliers=False, whis=1, palette="colorblind", order=order)
# a3 = sns.boxplot(ax=axs[2], data=all_properties, x="Cluster", y="q_r", showfliers=False, whis=1, palette="colorblind", order=order)
#
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="n_r", hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
# # a2 = sns.histplot(ax=axs[1], data=all_properties, x="pa_r", hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
# # a3 = sns.histplot(ax=axs[2], data=all_properties, x="q_r", hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
#
#
# # plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution_all_features")
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution_select_features")
# plt.show()




# # physical properties
# fig, axs = plt.subplots(2, 3, figsize=(30, 20))
#
# a1 = sns.boxplot(ax=axs[0, 0], data=all_properties, x="Cluster", y="re_r", showfliers=False, whis=1, palette="colorblind", order=order)
# a2 = sns.boxplot(ax=axs[0, 1], data=all_properties, x="Cluster", y="InitialMassWeightedStellarAge", showfliers=False, whis=1, palette="colorblind", order=order)
# a3 = sns.boxplot(ax=axs[0, 2], data=all_properties, x="Cluster", y="StarFormationRate", showfliers=False, whis=1, palette="colorblind", order=order)
#
# b1 = sns.boxplot(ax=axs[1, 0], data=all_properties, x="Cluster", y="MassType_Star", showfliers=False, whis=1, palette="colorblind", order=order)
# b2 = sns.boxplot(ax=axs[1, 1], data=all_properties, x="Cluster", y="MassType_DM", showfliers=False, whis=1, palette="colorblind", order=order)
# b3 = sns.boxplot(ax=axs[1, 2], data=all_properties, x="Cluster", y="MassType_BH", showfliers=False, whis=1, palette="colorblind", order=order)
#
#
# # plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution_all_features")
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution_select_features")
# plt.show()





# # number in elliptical cluster
# elliptical_cluster_count = all_properties[(all_properties["Cluster"] == 0)].shape[0]
#
# # number of ellipticals in elliptical cluster
# elliptical_in_elliptical = all_properties[((all_properties["Cluster"] == 0) & (all_properties["n_r"] <= 2.5))].shape[0]
#
# # number in spiral cluster
# spiral_cluster_count = all_properties[(all_properties["Cluster"] == 1)].shape[0]
#
# # number of spirals in spiral cluster
# spiral_in_spiral = all_properties[((all_properties["Cluster"] == 1) & (all_properties["n_r"] >= 2.5))].shape[0]
#
#
# print(spiral_in_spiral, "of", spiral_cluster_count, "spirals in spiral cluster", spiral_in_spiral/spiral_cluster_count)
# print(elliptical_in_elliptical, "of", elliptical_cluster_count, "ellipticals in elliptical cluster", elliptical_in_elliptical/elliptical_cluster_count)
#
#
# elliptical_count = all_properties[(all_properties["n_r"] >= 2.5)].shape[0]
# spiral_count = all_properties[(all_properties["n_r"] <= 2.5)].shape[0]
#
# print(elliptical_count)
# print(spiral_count)
#
#
# sns.histplot(data=all_properties, x="n_r", element="poly")
# plt.show()





# # sns.scatterplot(data=med_df, x=med_df["n_r"], y=np.log(med_df["MassType_Star"], hue="Cluster", palette="colorblind", s=75)
# sns.scatterplot(data=med_df, x=med_df["n_r"], y=np.log10(med_df["MassType_Star"]), s=75)
# # sns.scatterplot(data=med_df, x="n_r", y="MassType_Star", s=100)
#
# # sns.kdeplot(data=all_properties, x="n_r", y="MassType_Star", hue="Cluster", fill=True)
#
#
#
# # sns.scatterplot(data=all_properties, x="n_r", y="MassType_Star", alpha=0.3)
# # sns.scatterplot(data=all_properties, x="n_r", y="MassType_Star", alpha=0.3, hue="Cluster", palette="colorblind")
#
#
# # plt.scatter(x=all_properties["n_r"], y=all_properties["MassType_Star"], alpha=0.3, s=5)
# # plt.scatter(x=all_properties["n_r"], y=np.log10(all_properties["MassType_Star"]), alpha=1, s=5)
# # plt.hist2d(x=all_properties["n_r"], y=all_properties["MassType_Star"], bins=1000, cmap="Blues")
# # sns.kdeplot(data=all_properties, x=all_properties["n_r"], y=np.log10(all_properties["MassType_Star"]), fill=True, levels=100, cmap="Blues", thresh=0)
#
# plt.xlabel("Sersic Index")
# # plt.ylabel("Stellar Mass")
# plt.ylabel("Log(Stellar Mass)")
#
# plt.show()






print(med_df)
print()

order = med_df["n_r"].sort_values(ascending=False).index.to_list()
print(order)
print()


# for i, cluster in enumerate(order):
for cluster in range(0, n_clusters):


    galaxy_ids = all_properties[all_properties["Cluster"] == cluster]["GalaxyID"].tolist()
    sample = random.sample(galaxy_ids, 25)

    print(sample)

    # np.save("Clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(cluster) + ".npy", np.array(galaxy_ids[:25]))
    np.save("Clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(cluster) + ".npy", np.array(sample))


# 2 Clusters
# [9279688, 11546706, 11564243, 13296286, 8536493, 10088971, 15336992, 966292, 9747706, 17100831, 8986033, 9224716, 139400, 9478262, 9388126, 17592249, 12728081, 9126772, 16420095, 9469843, 16720414, 10487263, 17686442, 10806034, 9187925]
# [18359999, 13873825, 32367, 4518274, 9532695, 8686633, 9484896, 8216841, 9684480, 13698384, 11533908, 8128032, 16231498, 14653070, 16062295, 12701693, 8122788, 9920747, 3436085, 13715538, 10733159, 15328474, 16677355, 3523446, 8961773]

# 11 Clusters






print()

for cluster in range(0, n_clusters):
    print((cluster, all_properties[all_properties["Cluster"] == cluster].shape[0]))





# order = med_df["n_r"].sort_values(ascending=False).index.to_list()
#
# # print(order)
#
# fig, axs = plt.subplots(2, 5, figsize=(50, 20))
#
# count = 0
#
# for i in range(0, 2):
#     for j in range(0, 5):
#
#         sns.histplot(ax=axs[i, j], data=all_properties[all_properties["Cluster"] == order[count]], y="n_r", element="poly")
#
#         axs[i, j].set_ylim([0, 8])
#
#         count += 1
#
# plt.show()




