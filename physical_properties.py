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
encoding_dim = 44

# set the number of clusters
n_clusters = 12


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
print(med_physical_properties)
# print(med_structure_properties)



print(physical_properties.sort_values(by="Stellar_Mass/DM_Mass")[["GalaxyID", "MassType_DM", "MassType_Star", "Stellar_Mass/DM_Mass", "Cluster"]])



# sns.scatterplot(data=physical_properties, x="Log_Stellar_Mass", y="Stellar_Mass/DM_Mass", hue=structure_properties["n_r"])
# sns.scatterplot(data=med_physical_properties, x="Log_Stellar_Mass", y="Stellar_Mass/DM_Mass", hue=med_structure_properties["re_r"])
# plt.ylim(-0.1, 3)

# sns.histplot(data=physical_properties, x="Stellar_Mass/DM_Mass")
# sns.histplot(data=physical_properties, x="MassType_DM", bins=5000)
# sns.violinplot(data=physical_properties, y=["MassType_DM", "MassType_Star"])
# plt.ylim(-2e+12, 1.4e+13)
# plt.xlim(-0.1, 0.5)
# plt.ylim(-0.1, 5)
# plt.xlim(-0.001e+14, 0.2e+13)

# plt.scatter(physical_properties["MassType_Star"], physical_properties["MassType_Star"])

# plt.figure(figsize=(10, 8))

# # plt.scatter(x=med_physical_properties["Log_Stellar_Mass"], y=med_physical_properties["Stellar_Mass/DM_Mass"], c=med_structure_properties["n_r"], cmap="inferno_r", s=150, ec="black", lw=0.5)
plt.scatter(x=med_physical_properties["Log_DM_Mass"], y=med_physical_properties["Stellar_Mass/DM_Mass"], c=med_structure_properties["n_r"], cmap="inferno_r", s=150, ec="black", lw=0.5)


cbar = plt.colorbar()
cbar.set_label(label="Sersic Index", size=18, labelpad=20)
cbar.ax.tick_params(labelsize=12)


# plt.ylim(0.01, 0.05)
# plt.xlim(11.65, 12.3)

# plt.xlabel("Log(Stellar Mass)", fontsize=18, labelpad=20)
plt.xlabel("Log(Dark Matter Halo Mass)", fontsize=18, labelpad=20)
plt.ylabel("Stellar Mass / Dark Matter Halo Mass", fontsize=18, labelpad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("Plots/Stellar_Mass_DM_Mass_Sersic_crop")


plt.show()
