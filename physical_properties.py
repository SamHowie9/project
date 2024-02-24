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
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)




# set the encoding dimension (number of extracted features)
encoding_dim = 26

# set the number of clusters
n_clusters = 16


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


# load the physical properties and add the clusters
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv")
physical_properties.drop(physical_properties.tail(200).index, inplace=True)
physical_properties["Cluster"] = clusters

# load the structural measurements and add the clusters
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
structure_properties["Cluster"] = clusters

# print(structure_properties)



physical_properties["Log_Stellar_Mass"] = np.log10(physical_properties["MassType_Star"])
physical_properties["Log_DM_Mass"] = np.log10(physical_properties["MassType_DM"])
physical_properties["Stellar_Mass/DM_Mass"] = physical_properties["MassType_Star"]/physical_properties["MassType_DM"]


med_physical_properties = pd.DataFrame(columns=["MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate", "Log_Stellar_Mass", "Log_DM_Mass", "Stellar_Mass/DM_Mass"])
med_structure_properties = pd.DataFrame(columns=["n_r", "q_r", "re_r", "mag_r"])


for i in range(0, n_clusters):
    med_physical_cluster = physical_properties.loc[physical_properties["Cluster"] == i, ["MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate", "Log_Stellar_Mass", "Log_DM_Mass", "Stellar_Mass/DM_Mass"]].median()
    med_physical_properties.loc[i] = med_physical_cluster.tolist()

    med_structure_cluster = structure_properties.loc[structure_properties["Cluster"] == i, ["n_r", "q_r", "re_r", "mag_r"]].median()
    med_structure_properties.loc[i] = med_structure_cluster.tolist()

# print(med_physical_properties)





# print(physical_properties)
# print(med_physical_properties)
# print(med_structure_properties)



# print(physical_properties.sort_values(by="Stellar_Mass/DM_Mass")[["GalaxyID", "MassType_DM", "MassType_Star", "Stellar_Mass/DM_Mass", "Cluster"]])



# sns.scatterplot(data=physical_properties, x="Log_Stellar_Mass", y="Stellar_Mass/DM_Mass", hue=structure_properties["n_r"])
# sns.scatterplot(data=med_physical_properties, x="Log_Stellar_Mass", y="Stellar_Mass/DM_Mass", hue=med_structure_properties["re_r"])
# plt.ylim(-0.1, 3)

# sns.histplot(data=physical_properties, x="Stellar_Mass/DM_Mass")
# sns.histplot(data=physical_properties, x=np.log10(physical_properties["MassType_DM"]), bins=1000)
# sns.violinplot(data=physical_properties, y=["MassType_DM", "MassType_Star"])
# plt.ylim(-2e+12, 1.4e+13)
# plt.xlim(-0.1, 0.5)
# plt.ylim(-0.1, 5)
# plt.xlim(-0.001e+14, 0.2e+13)

# plt.scatter(physical_properties["MassType_Star"], physical_properties["MassType_Star"])

# plt.figure(figsize=(10, 8))

# # plt.scatter(x=med_physical_properties["Log_Stellar_Mass"], y=med_physical_properties["Stellar_Mass/DM_Mass"], c=med_structure_properties["n_r"], cmap="inferno_r", s=150, ec="black", lw=0.5)
# plt.scatter(x=med_physical_properties["Log_DM_Mass"], y=med_physical_properties["Stellar_Mass/DM_Mass"], c=med_structure_properties["n_r"], cmap="inferno_r", s=150, ec="black", lw=0.5)
#
#
# cbar = plt.colorbar()
# cbar.set_label(label="Sersic Index", size=18, labelpad=20)
# cbar.ax.tick_params(labelsize=12)


# plt.ylim(0.01, 0.05)
# plt.xlim(11.65, 12.3)

# plt.xlabel("Log(Stellar Mass)", fontsize=18, labelpad=20)
# plt.xlabel("Log(Dark Matter Halo Mass)", fontsize=18, labelpad=20)
# plt.ylabel("Stellar Mass / Dark Matter Halo Mass", fontsize=18, labelpad=20)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.savefig("Plots/Stellar_Mass_DM_Mass_Sersic_crop")
# plt.show()







# med_df = pd.merge(med_structure_properties, med_physical_properties, left_index=True, right_index=True)
# med_df["Cluster"] = list(range(0, n_clusters))
#
#
# ab_mag = med_df["mag_r"].sort_values(ascending=False).index.to_list()
# semi_major = med_df["re_r"].sort_values(ascending=False).index.to_list()
# sfr = med_df["StarFormationRate"].sort_values(ascending=False).index.to_list()
# stellar_mass = med_df["MassType_Star"].sort_values(ascending=False).index.to_list()
# dm_mass = med_df["MassType_DM"].sort_values(ascending=False).index.to_list()
# bh_mass = med_df["MassType_BH"].sort_values(ascending=False).index.to_list()
#
# fig, axs = plt.subplots(1, 3, figsize=(30, 8))
#
# b1 = sns.boxplot(ax=axs[0], data=structure_properties, x="Cluster", y="mag_r", showfliers=False, whis=1, palette="colorblind", order=ab_mag, hue="Cluster", dodge=False)
# b1.set_title("AB Magnitude", fontsize=20)
# b1.set_xlabel("Cluster", fontsize=15)
# b1.set_ylabel("AB Magnitude", fontsize=15)
# b1.tick_params(labelsize=12)
#
# b2 = sns.boxplot(ax=axs[1], data=structure_properties, x="Cluster", y="re_r", showfliers=False, whis=1, palette="colorblind", order=semi_major, hue="Cluster", dodge=False)
# b2.set_title("Semi-Major Axis", fontsize=20)
# b2.set_xlabel("Cluster", fontsize=15)
# b2.set_ylabel("Semi-Major Axis", fontsize=15)
# b2.tick_params(labelsize=12)
#
# b3 = sns.boxplot(ax=axs[2], data=physical_properties, x="Cluster", y="StarFormationRate", showfliers=False, whis=1, palette="colorblind", order=sfr, hue="Cluster", dodge=False)
# b3.set_title("Star Formation Rate", fontsize=20)
# b3.set_xlabel("Cluster", fontsize=15)
# b3.set_ylabel("Star Formation Rate", fontsize=15)
# b3.tick_params(labelsize=12)
#
# # b4 = sns.boxplot(ax=axs[0], data=physical_properties, x="Cluster", y="MassType_Star", showfliers=False, whis=1, palette="colorblind", order=stellar_mass, hue="Cluster", dodge=False)
# # b4.set_title("Stellar Mass", fontsize=20)
# # b4.set_xlabel("Cluster", fontsize=15)
# # b4.set_ylabel("Stellar Mass", fontsize=15)
# # b4.tick_params(labelsize=12)
# #
# # b5 = sns.boxplot(ax=axs[1], data=physical_properties, x="Cluster", y="MassType_DM", showfliers=False, whis=1, palette="colorblind", order=dm_mass, hue="Cluster", dodge=False)
# # b5.set_title("Dark Matter Mass", fontsize=20)
# # b5.set_xlabel("Cluster", fontsize=15)
# # b5.set_ylabel("Dark Matter Mass", fontsize=15)
# # b5.tick_params(labelsize=12)
# #
# # b6 = sns.boxplot(ax=axs[2], data=physical_properties, x="Cluster", y="MassType_BH", showfliers=False, whis=1, palette="colorblind", order=bh_mass, hue="Cluster", dodge=False)
# # b6.set_title("Black Hole Mass", fontsize=20)
# # b6.set_xlabel("Cluster", fontsize=15)
# # b6.set_ylabel("Black Hole Mass", fontsize=15)
# # b6.tick_params(labelsize=12)
#
# for ax in axs:
#     ax.legend([], [], frameon=False)
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_properties_half_1.png")
# plt.show()




print("High Sersic: ", physical_properties.loc[physical_properties["Cluster"] == 0]["GalaxyID"].tolist())
print("Low Sersic (Stripped): ",  physical_properties.loc[physical_properties["Cluster"] == 15]["GalaxyID"].tolist())
print("Stripped High Sersic:", physical_properties.loc[physical_properties["Cluster"] == 3]["GalaxyID"].tolist())

