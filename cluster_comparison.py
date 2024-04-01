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
# [33641, 5956833, 9289264, 7190132, 5370324, 7174790, 9762290, 8621306, 14949191, 9988503, 12656054, 14183869, 18343512, 18326938, 11500510, 11486871, 14487346, 9406027, 17229909, 4562581, 9677268, 10042584, 8920435, 18377512, 18106757]
# [13154469, 10107362, 15435358, 8991644, 11072600, 13263389, 12156652, 9883164, 17287247, 6642819, 9835510, 2424089, 5367150, 9144377, 10450102, 10027821, 14402768, 3550458, 16161069, 18329099, 12172318, 10153845, 10173307, 8770455, 16491889]
# [15242252, 9088199, 10586238, 8754568, 9884211, 10044250, 17057997, 11576383, 9057798, 17395890, 10468842, 9742670, 8395234, 8990039, 13785720, 8283638, 10487263, 10542488, 18111068, 8797511, 15827462, 12711010, 17285721, 12744882, 17552187]
# [8605415, 16116800, 18312498, 8791779, 234331, 16978215, 17858355, 10361791, 8331245, 8150447, 13659798, 16009894, 15451067, 10431234, 8900329, 9614142, 9604696, 2297047, 10694572, 9044113, 13220029, 14418321, 7678274, 17246198, 10753341]
# [17497367, 16525983, 15091775, 12867007, 9490638, 12654694, 16668993, 14737385, 8238669, 9470683, 8489085, 16001262, 17666085, 13212926, 8686633, 8743718, 10125259, 17323600, 11510905, 16550703, 9214339, 2506302, 9557493, 3435558, 17868998]
# [18135698, 13921561, 17805326, 18104018, 12096259, 17097594, 16895359, 3274716, 16921469, 14841935, 15597924, 15511664, 16536187, 17697185, 15274006, 17131844, 8747774, 14623009, 15978821, 9674774, 12731699, 8850936, 9052812, 16761385, 8266167]
# [8980187, 9709797, 9991035, 18416571, 8946831, 10251667, 135178, 43262, 9464627, 9269963, 16185809, 16604516, 9684480, 13667604, 18160043, 16789312, 16707700, 2293314, 14534193, 8292328, 17875912, 17986237, 8799478, 15157618, 12152678]
# [11456125, 16907882, 16715859, 17737062, 8886522, 13277309, 9808391, 11455096, 14371825, 10070092, 16723181, 10221113, 5348717, 2430739, 8264929, 8563943, 8976542, 18060726, 9858663, 8362460, 17775313, 5340165, 18197884, 8542194, 12782431]
# [8458138, 10326235, 9024232, 8711319, 10124104, 16190253, 9231886, 8476287, 9517737, 13919266, 10202983, 17851301, 12103691, 18292453, 9225418, 1363926, 13293366, 18269025, 10751314, 222927, 9642759, 8789334, 2062484, 8423930, 13160767]
# [2513876, 15595159, 10603735, 5990641, 9759527, 18342109, 9219351, 8707374, 13702204, 8779628, 13305071, 9309358, 8937440, 3545253, 18367234, 6602359, 8425845, 17969166, 15134529, 8615267, 4534539, 15424769, 10001176, 9319535, 9040800]
# [9384897, 9857344, 17763272, 8730963, 9015914, 8318303, 10653919, 9724896, 10334457, 16135805, 8819793, 8858658, 11457293, 9220513, 18390235, 17485459, 10738667, 9848106, 17921916, 15885743, 9362531, 18385553, 11658789, 15614725, 17704712]







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




