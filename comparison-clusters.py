import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestCentroid
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import image as mpimg
import random
import textwrap





pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)



# set the encoding dimension (number of extracted features)
encoding_dim = 20

run = 3

# set the number of clusters
n_clusters = 2







# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# # account for hte validation data and remove final 200 elements
# structure_properties.drop(structure_properties.tail(200).index, inplace=True)
# physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")


print(len(all_properties))


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()
# print(bad_fit)

# remove those galaxies
for i, galaxy in enumerate(bad_fit):
    all_properties = all_properties.drop(galaxy, axis=0)


print(len(all_properties))





# original dataset

# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/PCA/pca_features_" + str(encoding_dim) + "_features.npy")
# encoding_dim = extracted_features.shape[1]
# extracted_features_switch = extracted_features.T
#
#
# # # perform pca on the extracted features
# # pca = PCA(n_components=11).fit(extracted_features)
# # extracted_features = pca.transform(extracted_features)
# # extracted_features_switch = extracted_features.T
#
#
# # account for the training data in the dataframe
# all_properties = all_properties.drop(all_properties.tail(200).index, inplace=True)






# balanced dataset

# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# encoding_dim = extracted_features.shape[1]
# extracted_features_switch = extracted_features.T
#
# print(extracted_features.shape)
#
#
# # perform pca on the extracted features
# pca = PCA(n_components=13).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T
#
#
# # get the indices of the different types of galaxies (according to sersic index)
# spirals_indices = list(all_properties.loc[all_properties["n_r"] <= 2.5].index)
# unknown_indices = list(all_properties.loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index)
# ellipticals_indices = list(all_properties.loc[all_properties["n_r"] >= 4].index)
#
# # sample the galaxies to balance the dataset (as we did when training the model)
# random.seed(1)
# chosen_spiral_indices = random.sample(spirals_indices, round(len(spirals_indices)/2))
# chosen_ellipticals_indices = [index for index in ellipticals_indices for _ in range(4)]
# chosen_indices = chosen_spiral_indices + unknown_indices + chosen_ellipticals_indices
#
#
# # reorder the properties dataframe to match the extracted features of the balanced dataset
# all_properties = all_properties.loc[chosen_indices]
#
#
# # get the randomly sampled testing set indices
# random.seed(2)
# test_indices = random.sample(range(0, len(chosen_indices)), 20)
#
#
# # flag the training set in the properties dataframe (removing individually effects the position of the other elements)
# for i in test_indices:
#     all_properties.iloc[i] = np.nan
#
#
# # remove the training set from the properties dataframe
# all_properties = all_properties.dropna()





# fully balanced dataset

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

# load the extracted features
extracted_features = np.load("Variational Eagle/Extracted Features/Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
encoding_dim = extracted_features.shape[1]
extracted_features_switch = extracted_features.T

print(extracted_features.shape)

# extracted_features = extracted_features[:len(all_properties)]

print(extracted_features.shape)

# perform pca on the extracted features
pca = PCA(n_components=13).fit(extracted_features)
extracted_features = pca.transform(extracted_features)
# extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T








# perform binomial hierarchical clustering
binary_hierarchical = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
binary_clusters = binary_hierarchical.fit_predict(extracted_features)


binary_clusters = binary_clusters[:len(all_properties)]

# add the binomial cluster of each galaxy to properties dataframe
all_properties["Binary_Cluster"] = binary_clusters




# perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
clusters = hierarchical.fit_predict(extracted_features)

clusters = clusters[:len(all_properties)]

# add the cluster of each galaxy to the properties dataframe
all_properties["Cluster"] = clusters


all_properties["GalaxyID"] = all_properties["GalaxyID"].astype(int)

random.seed(6)

print(list(all_properties["GalaxyID"].loc[all_properties["Binary_Cluster"] == 0]))
print(list(all_properties["GalaxyID"].loc[all_properties["Binary_Cluster"] == 1]))
print()
print("Cluster 0:", random.sample(list(all_properties["GalaxyID"].loc[all_properties["Binary_Cluster"] == 0]), 16))
print("Cluster 1:", random.sample(list(all_properties["GalaxyID"].loc[all_properties["Binary_Cluster"] == 1]), 16))
print()
print()





# kmeans = KMeans(n_clusters=n_clusters)
# clusters = kmeans.fit_predict(chosen_features)

# spectral = SpectralClustering(n_clusters=n_clusters)
# clusters = spectral.fit_predict(chosen_features)

# hdbscan = HDBSCAN(metric="euclidean")
# clusters = hdbscan.fit_predict(chosen_features)


# # get hierarchical centers
# clf = NearestCentroid()
# clf.fit(extracted_features, clusters)
# centers = clf.centroids_
# centers_switch = centers.T
#
#
#
# mf = "#eec681"
# lf = "#7fb8d8"
#
# binary_palette = {0:mf, 1:mf, 2:lf, 3:lf, 4:lf, 5:mf, 6:mf, 7:mf, 8:mf, 9:mf, 10:mf, 11:mf, 12:mf, 13:lf}




# Apparent Magnitude, Stellar Mass, Semi-Major Axis, Sersic Index
# columns = ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]

columns = ["n_r", "pa_r", "q_r", "re_r", "mag_r", "flag_r", "MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate", "Binary_Cluster"]

med_df = pd.DataFrame(columns=columns)

for i in range(0, n_clusters):
    med_cluster = all_properties.loc[all_properties["Cluster"] == i, columns].median()
    med_df.loc[i] = med_cluster

med_df["Cluster"] = list(range(0, n_clusters))


# print(med_df.sort_values(by="n_r"))
# print()



plt.rc("text", usetex=True)



order_property = "n_r"

order = med_df[order_property].sort_values(ascending=False).index.to_list()


# # single property
# a1 = sns.boxplot(data=all_properties, x="Cluster", y="n_r", showfliers=False, whis=1, palette="colorblind", order=order)
# # a1 = sns.histplot(data=all_properties, x=property, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
#
# # plt.savefig("Plots/" + str(encoding_dim) + "_feature_3_" + str(n_clusters) + "_cluster_sersic_distribution_all_features")
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_3_" + str(n_clusters) + "_cluster_flag_distribution_select_features")
# # plt.savefig("Plots/" + str(encoding_dim) + "_feature_3_" + str(n_clusters) + "_cluster_sersic_distribution_select_features_hist")
# plt.show()




# # structure measurement box
# fig, axs = plt.subplots(3, 1, figsize=(20, 15))
#
# # a1 = sns.boxplot(ax=axs[0], data=all_properties, x="Cluster", y="n_r", showfliers=False, whis=1, palette=binary_palette, order=order)
# a1 = sns.boxplot(ax=axs[0], data=all_properties, x="Cluster", y="n_r", showfliers=False, whis=1, order=order)
# a1.set_ylabel("Sersic Index", fontsize=20)
# a1.set_xlabel(None)
# # mf_patch = mpatches.Patch(color=mf, label="More Featured")
# # lf_patch = mpatches.Patch(color=lf, label="Less Featured")
# # a1.legend(handles=[mf_patch, lf_patch], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
# # a1.set_xticks([1, 0], ["Less Featured", "More Featured"])
# # a1.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
# a1.tick_params(labelsize=20)
#
# # a2 = sns.boxplot(ax=axs[1], data=all_properties, x="Cluster", y=abs(all_properties["pa_r"]), showfliers=False, whis=1, palette=binary_palette, order=order)
# a2 = sns.boxplot(ax=axs[1], data=all_properties, x="Cluster", y=abs(all_properties["pa_r"]), showfliers=False, whis=1, order=order)
# a2.set_ylabel("Position Angle", fontsize=20)
# a2.set_xlabel(None)
# # a2.legend([],[], frameon=False)
# # a2.set_xticks([1, 0], ["Less Featured", "More Featured"])
# a2.tick_params(labelsize=20)
#
# # a3 = sns.boxplot(ax=axs[2], data=all_properties, x="Cluster", y="q_r", showfliers=False, whis=1, palette=binary_palette, order=order)
# a3 = sns.boxplot(ax=axs[2], data=all_properties, x="Cluster", y="q_r", showfliers=False, whis=1, order=order)
# a3.set_ylabel("Axis Ratio", fontsize=20)
# a3.set_xlabel("Group Index", fontsize=20, labelpad=10)
# # a3.set_xticks([1, 0], ["Less Featured", "More Featured"])
# # a3.legend([],[], frameon=False)
# a3.tick_params(labelsize=20)
#
# plt.savefig("Variational Eagle/Cluster Plots/vae_" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution_box", bbox_inches='tight')
# plt.show()







# struture measurement hist
fig, axs = plt.subplots(1, 3, figsize=(25, 5))

a1 = sns.histplot(ax=axs[0], data=all_properties, x="n_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
a1.set_ylabel("Normalised Frequency", fontsize=20)
a1.set_xlabel("Sersic Index", fontsize=20)
a1.set_yticks([])
a1.tick_params(labelsize=20)

a2 = sns.histplot(ax=axs[1], data=all_properties, x="pa_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
a2.set_ylabel("Normalised Frequency", fontsize=20)
a2.set_xlabel("Position Angle", fontsize=20)
a2.set_yticks([])
a2.set_xticks([-90, 0, 90], ["$-90^{\circ}$", "$0^{\circ}$", "$90^{\circ}$"])
a2.tick_params(labelsize=20)
a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})

a3 = sns.histplot(ax=axs[2], data=all_properties, x="q_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
a3.set_ylabel("Normalised Frequency", fontsize=20)
a3.set_xlabel("Axis Ratio", fontsize=20)
a3.set_yticks([])
a3.tick_params(labelsize=20)


# plt.savefig("Variational Eagle/Cluster Plots/" +  str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution", bbox_inches='tight')
plt.show()










# # disk = "#93ebe8"
# # bulge = "#ff9f9b"
# # machine_palette = {4:bulge, 3:bulge, 12:bulge, 13:bulge, 11:disk, 7:disk, 9:disk, 10:disk, 5:disk, 6:disk, 1:disk, 8:disk, 2:disk, 0:disk}
#
# # physical properties
# fig, axs = plt.subplots(3, 2, figsize=(20, 15))
#
# # a1 = sns.boxplot(ax=axs[0, 0], data=all_properties, x="Cluster", y="re_r", showfliers=False, whis=1, palette=machine_palette, order=med_df["re_r"].sort_values(ascending=False).index.to_list())
# a1 = sns.boxplot(ax=axs[0, 0], data=all_properties, x="Cluster", y="re_r", showfliers=False, whis=1, order=order)
# a1.set_ylabel("Semi-Major Axis (pkpk)", fontsize=20)
# a1.set_xlabel(None)
# a1.tick_params(labelsize=20)
#
# # a2 = sns.boxplot(ax=axs[1, 0], data=all_properties, x="Cluster", y="InitialMassWeightedStellarAge", showfliers=False, whis=1, palette=machine_palette, order=med_df["InitialMassWeightedStellarAge"].sort_values(ascending=False).index.to_list())
# a2 = sns.boxplot(ax=axs[1, 0], data=all_properties, x="Cluster", y="InitialMassWeightedStellarAge", showfliers=False, whis=1, order=order)
# a2.set_ylabel("Stellar Age (Gyr)", fontsize=20)
# a2.set_xlabel(None)
# a2.tick_params(labelsize=20)
#
# # a3 = sns.boxplot(ax=axs[2, 0], data=all_properties, x="Cluster", y="StarFormationRate", showfliers=False, whis=1, palette=machine_palette, order=med_df["StarFormationRate"].sort_values(ascending=False).index.to_list())
# a3 = sns.boxplot(ax=axs[2, 0], data=all_properties, x="Cluster", y="StarFormationRate", showfliers=False, whis=1, order=order)
# a3.set_ylabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
# a3.set_xlabel("Group Index", fontsize=20)
# a3.tick_params(labelsize=20)
#
#
# # b1 = sns.boxplot(ax=axs[0, 1], data=all_properties, x="Cluster", y=all_properties["MassType_Star"].div(1e10), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_Star"].sort_values(ascending=False).index.to_list())
# b1 = sns.boxplot(ax=axs[0, 1], data=all_properties, x="Cluster", y=all_properties["MassType_Star"].div(1e10), showfliers=False, whis=1, order=order)
# b1.set_ylabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
# b1.set_xlabel(None)
# b1.tick_params(labelsize=20)
# b1.set_yticks([0, 5, 10, 15])
# b1.set_ylim(-0.1, 15.2)
#
# # b2 = sns.boxplot(ax=axs[1, 1], data=all_properties, x="Cluster", y=all_properties["MassType_DM"].div(1e12), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_DM"].sort_values(ascending=False).index.to_list())
# b2 = sns.boxplot(ax=axs[1, 1], data=all_properties, x="Cluster", y=all_properties["MassType_DM"].div(1e12), showfliers=False, whis=1, order=order)
# b2.set_ylabel("Dark Matter Mass ($10^{12}$M$_{\odot}$)", fontsize=20)
# b2.set_xlabel(None)
# b2.tick_params(labelsize=20)
#
# # b3 = sns.boxplot(ax=axs[2, 1], data=all_properties, x="Cluster", y=all_properties["MassType_BH"].div(1e8), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_BH"].sort_values(ascending=False).index.to_list())
# b3 = sns.boxplot(ax=axs[2, 1], data=all_properties, x="Cluster", y=all_properties["MassType_BH"].div(1e8), showfliers=False, whis=1, order=order)
# b3.set_ylabel("Black Hole Mass ($10^{8}$M$_{\odot}$)", fontsize=20)
# b3.set_xlabel("Group Index", fontsize=20)
# b3.tick_params(labelsize=20)
#
# # disk_patch = mpatches.Patch(color=disk, label="Disk Structures")
# # bulge_patch = mpatches.Patch(color=bulge, label="Bulge Structures")
# # a1.legend(handles=[disk_patch, bulge_patch], bbox_to_anchor=(0.52, 0.93), loc='upper center', bbox_transform=fig.transFigure, ncol=2, prop={"size":20})
#
# # plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution_all_features")
# plt.savefig("Variational Eagle/Cluster Plots/vae_" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution", bbox_inches='tight')
# # plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution_sersic_order", bbox_inches='tight')
# plt.show()






# all physical histogram
# fig, axs = plt.subplots(3, 2, figsize=(20, 18))
#
# bins = np.histogram_bin_edges(all_properties["re_r"], bins=20)
# combined_bins = np.sum(np.histogram(all_properties["re_r"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["re_r"], bins=bins[:20])[0], [combined_bins]))
#
# a1 = sns.histplot(ax=axs[0, 0], data=all_properties, x="re_r", bins=new_bins, stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0])
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="re_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# a1.set_xlabel("Semi-Major Axis (pkpc)", fontsize=20)
# a1.set_yticks([])
# a1.tick_params(labelsize=20)
# a1.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0.52, 0.93), loc='upper center', bbox_transform=fig.transFigure, ncol=2, prop={"size":20})
#
#
# a2 = sns.histplot(ax=axs[1, 0], data=all_properties, x="InitialMassWeightedStellarAge", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# a2.set_xlabel("Stellar Age (Gyr)", fontsize=20)
# a2.set_yticks([])
# a2.tick_params(labelsize=20)
#
# a3 = sns.histplot(ax=axs[2, 0], data=all_properties, x="StarFormationRate", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=50, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# # a3.set_xlabel("Star Formation Rate ($\mathrm{M}_{\odot}\mathrm{yr}^{-1}$)", fontsize=20)
# a3.set_xlabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
#
# a3.set_yticks([])
# a3.set_xlim(0, 6)
# a3.tick_params(labelsize=20)
#
#
#
# stellar_mass = all_properties["MassType_Star"].div(1e10)
#
# bins = np.histogram_bin_edges(stellar_mass, bins=100)
# combined_bins = np.sum(np.histogram(stellar_mass, bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(stellar_mass, bins=bins[:20])[0], [combined_bins]))
#
# a1 = sns.histplot(ax=axs[0, 1], data=all_properties, x=stellar_mass, bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="MassType_Star", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=50, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# # a1.set_xlabel("Stellar Mass ($10^{10} \mathrm{M}_{\odot}$)", fontsize=20)
# a1.set_xlabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
# a1.set_yticks([])
# a1.set_xticks([0, 5, 10])
# a1.tick_params(labelsize=20)
#
#
#
# dm_mass = all_properties["MassType_DM"].div(1e12)
#
# bins = np.histogram_bin_edges(dm_mass, bins=250)
# combined_bins = np.sum(np.histogram(dm_mass, bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(dm_mass, bins=bins[:20])[0], [combined_bins]))
#
# a2 = sns.histplot(ax=axs[1, 1], data=all_properties, x=dm_mass, bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a2 = sns.histplot(ax=axs[1], data=all_properties, x="MassType_DM", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=170)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# # a2.set_xlabel("Dark Matter Mass ($10^{12} \mathrm{M}_{\odot}$)", fontsize=20)
# a2.set_xlabel("Dark Matter Mass ($10^{12}$M$_{\odot}$)", fontsize=20)
# a2.set_yticks([])
# a2.set_xticks([0, 5, 10])
# a2.tick_params(labelsize=20)
# # a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
#
#
# bh_mass = all_properties["MassType_BH"].div(1e8)
#
# bins = np.histogram_bin_edges(bh_mass, bins=250)
# combined_bins = np.sum(np.histogram(bh_mass, bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(bh_mass, bins=bins[:20])[0], [combined_bins]))
#
# a3 = sns.histplot(ax=axs[2, 1], data=all_properties, x=bh_mass, bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a3 = sns.histplot(ax=axs[2], data=all_properties, x="MassType_BH", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=150, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# # a3.set_xlabel("Black Hole Mass ($10^{8} \mathrm{M}_{\odot}$)", fontsize=20)
# a3.set_xlabel("Black Hole Mass ($10^{8}$M$_{\odot}$)", fontsize=20)
# a3.set_yticks([])
# # a3.set_xlim(0, 0.5e9)
# a3.tick_params(labelsize=20)
#
# # plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_all_physical_distribution", bbox_inches='tight')
# plt.show()




# # physical properties histogram
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
# bins = np.histogram_bin_edges(all_properties["re_r"], bins=20)
# combined_bins = np.sum(np.histogram(all_properties["re_r"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["re_r"], bins=bins[:20])[0], [combined_bins]))
#
# a1 = sns.histplot(ax=axs[0], data=all_properties, x="re_r", bins=new_bins, stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], legend=False)
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="re_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# a1.set_xlabel("Semi-Major Axis", fontsize=20)
# a1.set_yticks([])
# a1.tick_params(labelsize=20)
#
# a2 = sns.histplot(ax=axs[1], data=all_properties, x="InitialMassWeightedStellarAge", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# a2.set_xlabel("Stellar Age", fontsize=20)
# a2.set_yticks([])
# a2.tick_params(labelsize=20)
# a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
#
# a3 = sns.histplot(ax=axs[2], data=all_properties, x="StarFormationRate", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=50, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# a3.set_xlabel("Star Formation Rate", fontsize=20)
# a3.set_yticks([])
# a3.set_xlim(0, 6)
# a3.tick_params(labelsize=20)
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution", bbox_inches='tight')
# plt.show()



# # mass histogram
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
# bins = np.histogram_bin_edges(all_properties["MassType_Star"], bins=100)
# combined_bins = np.sum(np.histogram(all_properties["MassType_Star"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["MassType_Star"], bins=bins[:20])[0], [combined_bins]))
#
# a1 = sns.histplot(ax=axs[0], data=all_properties, x="MassType_Star", bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="MassType_Star", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=50, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# a1.set_xlabel("Stellar Mass", fontsize=20)
# a1.set_yticks([])
# # a1.set_xlim(0, 2e11)
# a1.tick_params(labelsize=20)
#
#
# bins = np.histogram_bin_edges(all_properties["MassType_DM"], bins=250)
# combined_bins = np.sum(np.histogram(all_properties["MassType_DM"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["MassType_DM"], bins=bins[:20])[0], [combined_bins]))
#
# a2 = sns.histplot(ax=axs[1], data=all_properties, x="MassType_DM", bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0])
# # a2 = sns.histplot(ax=axs[1], data=all_properties, x="MassType_DM", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=170)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# a2.set_xlabel("Dark Matter Mass", fontsize=20)
# a2.set_yticks([])
# # a2.set_xlim(0, 1.2e13)
# a2.tick_params(labelsize=20)
# a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
#
#
# bins = np.histogram_bin_edges(all_properties["MassType_BH"], bins=250)
# combined_bins = np.sum(np.histogram(all_properties["MassType_BH"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["MassType_BH"], bins=bins[:20])[0], [combined_bins]))
#
# a3 = sns.histplot(ax=axs[2], data=all_properties, x="MassType_BH", bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a3 = sns.histplot(ax=axs[2], data=all_properties, x="MassType_BH", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=150, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# a3.set_xlabel("Black Hole Mass", fontsize=20)
# a3.set_yticks([])
# # a3.set_xlim(0, 0.5e9)
# a3.tick_params(labelsize=20)
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_mass_distribution", bbox_inches='tight')
# plt.show()



total = all_properties.shape[0]
total_spiral = all_properties[all_properties["n_r"] <= 2.5].shape[0]
total_unknown = all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")].shape[0]
total_elliptical = all_properties[all_properties["n_r"] >= 4].shape[0]

print(str(total_spiral) + " Spirals (" + str(total_spiral/total) + ")")
print(str(total_unknown) + " Unknown (" + str(total_unknown/total) + ")")
print(str(total_elliptical) + " Ellipticals (" + str(total_elliptical/total) + ")")
print()


# total_4 = all_properties[all_properties["Cluster"] == 4].shape[0]
# total_4_spiral = all_properties[((all_properties["Cluster"] == 4) & (all_properties["n_r"] <= 2.5))].shape[0]
# total_4_elliptical = all_properties[((all_properties["Cluster"] == 4) & (all_properties["n_r"] > 2.5))].shape[0]
#
# print("Cluster 4 - " + str(total_4))
# print("Cluster 4 - " + str(total_4_spiral) + " Spirals (" + str(total_4_spiral/total_4) + ")")
# print("Cluster 4 - " + str(total_4_elliptical) + " Ellipticals (" + str(total_4_elliptical/total_4) + ")")
# print()



order = med_df["n_r"].sort_values().index.to_list()

for i in order:
    total_i = all_properties[all_properties["Cluster"] == i].shape[0]
    total_i_spiral = all_properties[((all_properties["Cluster"] == i) & (all_properties["n_r"] <= 2.5))].shape[0]
    total_i_unknown = all_properties[((all_properties["Cluster"] == i) & (all_properties["n_r"].between(2.5, 4, inclusive="neither")))].shape[0]
    total_i_elliptical = all_properties[((all_properties["Cluster"] == i) & (all_properties["n_r"] >= 4))].shape[0]

    print("Cluster " + str(i) + " - " + str(total_i))
    print("Cluster " + str(i) + " - " + str(total_i_spiral) + " Spirals (" + str(total_i_spiral / total_i) + ")")
    print("Cluster " + str(i) + " - " + str(total_i_unknown) + " Unknown (" + str(total_i_unknown / total_i) + ")")
    print("Cluster " + str(i) + " - " + str(total_i_elliptical) + " Ellipticals (" + str(total_i_elliptical / total_i) + ")")
    print()



# total_1 = all_properties[all_properties["Cluster"] == 1].shape[0]
# total_1_spiral = all_properties[((all_properties["Cluster"] == 1) & (all_properties["n_r"] <= 2.5))].shape[0]
# total_1_elliptical = all_properties[((all_properties["Cluster"] == 1) & (all_properties["n_r"] > 2.5))].shape[0]
#
# print("Cluster 1 - " + str(total_1))
# print("Cluster 1 - " + str(total_1_spiral) + " Spirals (" + str(total_1_spiral/total_1) + ")")
# print("Cluster 1 - " + str(total_1_elliptical) + " Ellipticals (" + str(total_1_elliptical/total_1) + ")")
# print()


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





# types = ["Spirals", "Spirals", "Spirals", "Ellipticals", "Ellipticals", "Spirals", "Spirals", "Spirals", "Spirals", "Spirals", "Spirals", "Spirals", "Barred Spirals", "Ellipticals"]
# med_df["Type"] = types
#
# print(sns.color_palette("colorblind").as_hex())
#
# spiral = "#01a5fe"
# bar = "#029e73"
# elliptical = "#fbb337"
# # triple_palette = {4:elliptical, 3:elliptical, 12:bar, 13:elliptical, 11:spiral, 7:spiral, 9:spiral, 10:spiral, 5:spiral, 6:spiral, 1:spiral, 8:spiral, 2:spiral, 0:spiral}
# triple_palette = [spiral, bar, elliptical]
#
#
# fig, axs = plt.subplots(2, 3, figsize=(25, 12))
#
#
# sm_s = sns.scatterplot(ax=axs[0, 0], data=med_df, x="n_r", y=med_df["MassType_Star"].div(1e10), hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sm_s.set_xlabel("Sersic Index", fontsize=20)
# sm_s.set_ylabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
#
# sm_s.tick_params(labelsize=20)
#
#
# sa_s = sns.scatterplot(ax=axs[0, 1], data=med_df, x="n_r", y="InitialMassWeightedStellarAge", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150)
#
# sa_s.set_xlabel("Sersic Index", fontsize=20)
# sa_s.set_ylabel("Stellar Age (Gyr)", fontsize=20)
# sa_s.tick_params(labelsize=20)
#
#
# sm_sa = sns.scatterplot(ax=axs[0, 2], data=med_df, x="InitialMassWeightedStellarAge", y=med_df["MassType_Star"].div(1e10), hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sm_sa.set_xlabel("Stellar Age (Gyr)", fontsize=20)
# sm_sa.set_ylabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
# sm_sa.tick_params(labelsize=20)
#
#
#
# # sfr_s = sns.scatterplot(ax=axs[0, 1], data=med_df, x="n_r", y="StarFormationRate", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150)
# #
# # sfr_s.set_xlabel("Sersic Index", fontsize=20)
# # sfr_s.set_ylabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
# # sfr_s.tick_params(labelsize=20)
#
#
#
# sma_s = sns.scatterplot(ax=axs[1, 0], data=med_df, x="n_r", y="re_r", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sma_s.set_xlabel("Sersic Index", fontsize=20)
# sma_s.set_ylabel("Semi-Major Axis (pkpc)", fontsize=20)
# sma_s.tick_params(labelsize=20)
#
# sfr_s = sns.scatterplot(ax=axs[1, 1], data=med_df, x="n_r", y="StarFormationRate", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sfr_s.set_xlabel("Sersic Index", fontsize=20)
# sfr_s.set_ylabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
# sfr_s.tick_params(labelsize=20)
#
#
# sma_sfr = sns.scatterplot(ax=axs[1, 2], data=med_df, x="StarFormationRate", y="re_r", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sma_sfr.set_xlabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
# sma_sfr.set_ylabel("Semi-Major Axis (pkpc)", fontsize=20)
# sma_sfr.tick_params(labelsize=20)
#
#
# sa_s.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", ncol=3, prop={"size":20})
# # a.legend(bbox_to_anchor=(0.52, 0.93), loc='upper center', bbox_transform=fig.transFigure, ncol=3, prop={"size":20})
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_sersic_vs_physical", bbox_inches='tight')
# plt.show()






# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
#
# sm = sns.scatterplot(ax=axs[0], data=med_df, x=med_df["MassType_Star"].div(1e10), y="re_r", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# sm.set_xlabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
# sm.set_ylabel("Semi-Major Axis", fontsize=20)
# sm.tick_params(labelsize=20)
#
#
# dm = sns.scatterplot(ax=axs[1], data=med_df, x=med_df["MassType_DM"].div(1e12), y="re_r", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150)
#
# dm.set_xlabel("Dark Matter Mass ($10^{12}$M$_{\odot}$)", fontsize=20)
# dm.set_ylabel("Semi-Major Axis", fontsize=20)
# dm.tick_params(labelsize=20)
#
#
# bh = sns.scatterplot(ax=axs[2], data=med_df, x=med_df["MassType_BH"].div(1e8), y="re_r", hue="Type", hue_order=["Spirals", "Barred Spirals", "Ellipticals"], palette=triple_palette, s=150, legend=False)
#
# bh.set_xlabel("Black Hole Mass ($10^{8}$M$_{\odot}$)", fontsize=20)
# bh.set_ylabel("Semi-Major Axis", fontsize=20)
# bh.tick_params(labelsize=20)
#
#
#
# dm.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", ncol=3, prop={"size":20})
# # a.legend(bbox_to_anchor=(0.52, 0.93), loc='upper center', bbox_transform=fig.transFigure, ncol=3, prop={"size":20})
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_semi-major_vs_mass", bbox_inches='tight')
# plt.show()








# spiral = "#01a5fe"
# bar = "#029e73"
# elliptical = "#fbb337"
#
# bl = "blue"
# r = "red"
# y = "yellow"
# b = "black"
# w = "white"
#
# triple_palette = {4:elliptical, 3:elliptical, 12:bar, 13:elliptical, 11:spiral, 7:spiral, 9:spiral, 10:spiral, 5:spiral, 6:spiral, 1:spiral, 8:spiral, 2:spiral, 0:spiral}
# triple_palette = {4:w, 3:r, 12:w, 13:y, 11:w, 7:b, 9:w, 10:w, 5:w, 6:w, 1:w, 8:bl, 2:w, 0:w}
#
#
#
# plt.figure(figsize=(20, 20))
#
# a = sns.scatterplot(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Cluster"], hue_order=[4, 3, 12, 13, 11, 7, 9, 10, 5, 6, 1, 8, 2, 0], palette=triple_palette, s=30)
# # a = sns.scatterplot(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Cluster"], palette="colorblind", s=30)
# # a = sns.scatterplot(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Cluster"], hue_order=med_df["MassType_Star"].sort_values(ascending=False).index.to_list(), palette="Blues", s=30)
#
# # plt.scatter(x=extracted_features_switch[7], y=all_properties["re_r"], hue=all_properties["Semi-Major Type"], s=3, alpha=0.5)
# # plt.plot(x_fit_3, y_fit_3, c="black")
# # a.set_ylim([0, 10])
# # a.set_xlim([-15, -3])
# a.set_xlabel("Feature 7", fontsize=20)
# a.set_ylabel("Semi-Major Axis", fontsize=20)
# a.tick_params(labelsize=20)
#
# plt.show()






# print(med_df)
# print()
#
# order = med_df["n_r"].sort_values(ascending=False).index.to_list()
# print(order)
# print()




# # for i, cluster in enumerate(order):
# for cluster in range(0, n_clusters):
#
#
#     galaxy_ids = all_properties[all_properties["Cluster"] == cluster]["GalaxyID"].tolist()
#     sample = random.sample(galaxy_ids, 25)
#
#     print(sample)
#
#     # np.save("Clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(cluster) + ".npy", np.array(galaxy_ids[:25]))
#     np.save("Clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(cluster) + ".npy", np.array(sample))
#


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

# 9 Clusters
# [17316116, 10567567, 18096816, 18230284, 9164674, 13859488, 17845684, 9176118, 8903544, 9848106, 10092169, 10193361, 17379782, 8111398, 15827462, 8391631, 16517497, 17213206, 9806843, 9615539, 14202038, 10391043, 9658285, 18000128, 16472250]
# [13848162, 15457920, 9189661, 17835495, 17948664, 10159085, 14115666, 17868998, 8483646, 10479082, 12785592, 16664982, 9214339, 3538779, 13825637, 15927500, 13700898, 15386755, 13156706, 10495333, 9256846, 9938601, 9133843, 13683867, 16750450]
# [10467986, 12711010, 3539991, 10344939, 2948351, 9011155, 8894369, 9201926, 8670585, 8269571, 9368071, 10187399, 17691609, 17509947, 3458705, 12659357, 13773122, 12654694, 8791779, 16475748, 13851369, 13688474, 13809907, 9683063, 8472770]
# [17853027, 14109686, 2304937, 17242991, 13215580, 11318263, 16647246, 10080502, 10195407, 13935855, 10545920, 7182472, 7629051, 17369519, 8503366, 10161161, 10108400, 8638279, 12167142, 9336486, 15964287, 7653318, 234331, 2804983, 9059235]
# [8139480, 11106875, 18043221, 2658206, 17886931, 9817539, 11568392, 18051512, 138061, 9668254, 10890785, 16565965, 17585533, 18387573, 13639821, 17866289, 14449268, 4209798, 13945982, 16668993, 11361114, 10149790, 10651124, 17871232, 10058550]
# [13195729, 9181451, 9268384, 16244165, 8505438, 14974620, 10408026, 8406315, 8624088, 9639907, 10868197, 17307029, 9026380, 13640612, 9627417, 10399381, 17981471, 13796727, 8649269, 16467494, 17339025, 9987253, 10776576, 9759527, 127772]
# [11559220, 9532695, 15601461, 3537758, 8789334, 13293366, 9872799, 9955910, 9410625, 9435793, 4580964, 12129529, 2631022, 9455775, 5309323, 65696, 3528962, 10038409, 8652246, 14384640, 8131927, 5959322, 9517737, 965536, 10939324]
# [10001965, 8850936, 15535362, 17464778, 12198852, 16268672, 9446777, 9537912, 12715080, 8274107, 15973150, 17363053, 16150066, 11419698, 14623009, 16403899, 18143519, 16736005, 17483370, 18135698, 15367781, 6066836, 18215143, 14841935, 15250310]
# [17357223, 8729072, 17795988, 8409334, 18359999, 14308492, 8358441, 8292328, 14607788, 8799478, 14895219, 2663001, 14459774, 8071906, 9284260, 10620766, 8206768, 16561331, 9508203, 15157618, 16281215, 13873825, 17432571, 17523255, 11448053]

# 14 Cluters
# [17462825, 10953355, 7659938, 4456330, 8545514, 11048307, 11072600, 13709148, 2659780, 10174638, 11118588, 15996483, 10545920, 13869651, 2062484, 18016380, 9869936, 6615741, 1738146, 17568884, 10455657, 6022578, 13637023, 1732243, 8141418]
# [8456338, 13190588, 13207800, 16996938, 8696150, 16891181, 8132671, 17704712, 9842936, 12181022, 17154752, 6086970, 18271269, 9937506, 9295169, 18188055, 10120919, 10339665, 8918814, 13793734, 16135805, 9662332, 9364039, 10262736, 10311974]
# [8279116, 12746617, 12760724, 17691609, 2760302, 16986868, 6600556, 9361157, 12089856, 16123379, 8525643, 8117170, 13864314, 12673208, 10750094, 11506546, 10223986, 12711010, 8628414, 14770895, 13809907, 9473362, 16231498, 9133843, 11893576]
# [11495444, 8133712, 6082967, 10125259, 9709797, 17445074, 14653070, 2490438, 9158108, 9561734, 10905233, 7680777, 12748608, 10282281, 18054058, 9484896, 11176936, 9481539, 17723839, 9189661, 1703010, 10196456, 17317486, 2065457, 15751379]
# [15973150, 8206768, 9446777, 17068442, 16268672, 8850936, 14681035, 16608247, 16878001, 17464778, 15046949, 9455775, 6066836, 8915064, 23302, 8968984, 18138348, 15402076, 15328474, 9192917, 15597924, 16204628, 8997131, 9578505, 15635733]
# [15180625, 10644457, 9623845, 13250441, 10766286, 10345837, 17731119, 16981992, 10128773, 14774114, 9804192, 2116222, 4521742, 3473902, 2618225, 15641198, 9565394, 17418548, 9045697, 9344085, 14696152, 8471989, 12105972, 9775887, 8883149]
# [9851671, 8854395, 18408011, 8331245, 14674761, 17592249, 4566939, 7645674, 8323311, 17906045, 9831288, 16005099, 2603883, 9752527, 10508211, 10425479, 7613426, 4580964, 10258829, 16030140, 9779253, 8115174, 16238798, 17651636, 9683063]
# [56966, 16902854, 10701959, 6026428, 14007136, 4471814, 12742443, 2721237, 16079259, 9794573, 11482234, 12641632, 14183869, 8953149, 1390573, 11557929, 8435854, 8109571, 14021166, 14822490, 11476544, 18120450, 15189924, 4522287, 14809907]
# [13722616, 10006224, 64010, 8929122, 16167926, 10225164, 12178298, 17275742, 15595159, 17664411, 15651343, 9147203, 8809752, 10669399, 10222240, 9296341, 8752042, 10697455, 10229106, 10776576, 10564851, 8814843, 10375596, 9747706, 18070830]
# [11556600, 16521446, 10934378, 9618714, 16472250, 8776660, 8653535, 9961051, 8423930, 10568231, 9620151, 9738646, 10092169, 13160767, 16517497, 9753783, 16539731, 8648068, 17746163, 10437671, 17025268, 15242252, 9966101, 9550846, 10029007]
# [11490586, 223470, 8882043, 18410129, 17228327, 3544703, 8349164, 9865665, 18240265, 10279009, 8460064, 11079170, 9517737, 9763692, 9331416, 8789334, 15172289, 8364907, 9632386, 14211587, 16647246, 14861046, 13684844, 13183667, 6631027]
# [9733376, 9858663, 10664750, 13719119, 10319358, 8667712, 9051081, 9813652, 12663357, 16275866, 9308022, 16459279, 15018060, 10565599, 18340514, 13221593, 1385469, 1784921, 9887869, 13698384, 17436625, 10388775, 615891, 10200519, 17331332]
# [8120452, 14476589, 16140911, 10323277, 15528598, 16627260, 9067175, 3456596, 18129275, 16024915, 10201809, 14143266, 9782789, 10890785, 9406027, 9334220, 9958489, 1407376, 14761477, 10211485, 9612643, 14357285, 17283836, 14648182, 9748723]
# [14349778, 15554596, 17467744, 10604961, 13151896, 9684480, 8216841, 16604516, 9556291, 8092294, 1017877, 16012583, 14188900, 8993208, 8090113, 12715080, 16284846, 7658846, 12122358, 17875912, 3524790, 10690127, 8733179, 40905, 8591424]






# print()
#
# for cluster in range(0, n_clusters):
#     print((cluster, all_properties[all_properties["Cluster"] == cluster].shape[0]))





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




