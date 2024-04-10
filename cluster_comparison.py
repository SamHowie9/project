import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
from sklearn.neighbors import NearestCentroid
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import image as mpimg
import random
import textwrap


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)



# set the encoding dimension (number of extracted features)
encoding_dim = 38

# set the number of clusters
n_clusters = 14







# # optimal clusters
# model = AgglomerativeClustering()
# visualizer = KElbowVisualizer(model, k=(2, 20))
# visualizer.fit(chosen_features)
# visualizer.show()






# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")









# for i in range(0, 5):
#     print(all_properties[(all_properties["flag_r"] == i)].shape[0])
#
#
# print((1, all_properties[(all_properties["flag_r"] == 1)]["GalaxyID"].tolist()))
# print((2, all_properties[(all_properties["flag_r"] == 2)]["GalaxyID"].tolist()))
# print((4, all_properties[(all_properties["flag_r"] == 4)]["GalaxyID"].tolist()))
#
# print(all_properties[(all_properties["flag_r"] == 5)])

# print(all_properties.sort_values("n_r", ascending=False))



# load the extracted features
extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_3.npy")


bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

for i, galaxy in enumerate(bad_fit):
    extracted_features = np.delete(extracted_features, galaxy-i, 0)
    all_properties = all_properties.drop(galaxy, axis=0)


extracted_features_switch = extracted_features.T


# chose which features to use for clustering
# meaningful_features = [8, 11, 12, 13, 14, 15, 16, 18, 20, 21]   # 24
# meaningful_features = [1, 2, 7, 10, 16, 20, 23, 27, 29, 36]  # 19
meaningful_features = [1, 2, 3, 4, 7, 8, 12, 20, 24, 26, 28]  # 26
# meaningful_features = [2, 3, 4, 7, 12, 20, 24, 26, 28]

chosen_features = []

for feature in meaningful_features:
    chosen_features.append(list(extracted_features_switch[feature]))

chosen_features = np.array(chosen_features).T



# chosen_features = extracted_features


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(chosen_features)

all_properties["Cluster"] = clusters


binary_hierarchical = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
binary_clusters = binary_hierarchical.fit_predict(chosen_features)
all_properties["Binary_Cluster"] = binary_clusters


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
centers_switch = centers.T



mf = "#eec681"
lf = "#7fb8d8"

binary_palette = {0:mf, 1:mf, 2:lf, 3:lf, 4:lf, 5:mf, 6:mf, 7:mf, 8:mf, 9:mf, 10:mf, 11:mf, 12:mf, 13:lf}




# Apparent Magnitude, Stellar Mass, Semi-Major Axis, Sersic Index
# columns = ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]

columns = ["n_r", "pa_r", "q_r", "re_r", "mag_r", "flag_r", "MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate", "Binary_Cluster"]

med_df = pd.DataFrame(columns=columns)

for i in range(0, n_clusters):
    med_cluster = all_properties.loc[all_properties["Cluster"] == i, columns].median()
    med_df.loc[i] = med_cluster

med_df["Cluster"] = list(range(0, n_clusters))


print(med_df)



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
# a1 = sns.boxplot(ax=axs[0], data=all_properties, x="Cluster", y="n_r", showfliers=False, whis=1, palette=binary_palette, order=order)
# a1.set_ylabel("Sersic Index", fontsize=20)
# a1.set_xlabel(None)
# mf_patch = mpatches.Patch(color=mf, label="More Featured")
# lf_patch = mpatches.Patch(color=lf, label="Less Featured")
# a1.legend(handles=[mf_patch, lf_patch], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
# # a1.set_xticks([1, 0], ["Less Featured", "More Featured"])
# # a1.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
# a1.tick_params(labelsize=20)
#
# a2 = sns.boxplot(ax=axs[1], data=all_properties, x="Cluster", y=abs(all_properties["pa_r"]), showfliers=False, whis=1, palette=binary_palette, order=order)
# a2.set_ylabel("Position Angle", fontsize=20)
# a2.set_xlabel(None)
# a2.legend([],[], frameon=False)
# # a2.set_xticks([1, 0], ["Less Featured", "More Featured"])
# a2.tick_params(labelsize=20)
#
# a3 = sns.boxplot(ax=axs[2], data=all_properties, x="Cluster", y="q_r", showfliers=False, whis=1, palette=binary_palette, order=order)
# a3.set_ylabel("Axis Ratio", fontsize=20)
# a3.set_xlabel(None)
# # a3.set_xticks([1, 0], ["Less Featured", "More Featured"])
# a3.legend([],[], frameon=False)
# a3.tick_params(labelsize=20)
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution_box", bbox_inches='tight')
# plt.show()







# # struture measurement hist
# fig, axs = plt.subplots(1, 3, figsize=(25, 5))
#
# a1 = sns.histplot(ax=axs[0], data=all_properties, x="n_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# a1.set_xlabel("Sersic Index", fontsize=20)
# a1.set_yticks([])
# a1.tick_params(labelsize=20)
#
# a2 = sns.histplot(ax=axs[1], data=all_properties, x="pa_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# a2.set_xlabel("Position Angle", fontsize=20)
# a2.set_yticks([])
# a2.set_xticks([-90, 0, 90], ["$-90^{\circ}$", "$0^{\circ}$", "$90^{\circ}$"])
# a2.tick_params(labelsize=20)
# a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
#
# a3 = sns.histplot(ax=axs[2], data=all_properties, x="q_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# a3.set_xlabel("Axis Ratio", fontsize=20)
# a3.set_yticks([])
# a3.tick_params(labelsize=20)
#
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_structure_distribution", bbox_inches='tight')
# plt.show()





disk = "#93ebe8"
bulge = "#ff9f9b"
machine_palette = {4:bulge, 3:bulge, 12:bulge, 13:bulge, 11:disk, 7:disk, 9:disk, 10:disk, 5:disk, 6:disk, 1:disk, 8:disk, 2:disk, 0:disk}

# physical properties
fig, axs = plt.subplots(3, 2, figsize=(20, 15))

a1 = sns.boxplot(ax=axs[0, 0], data=all_properties, x="Cluster", y="re_r", showfliers=False, whis=1, palette=machine_palette, order=med_df["re_r"].sort_values(ascending=False).index.to_list())
a1.set_ylabel("Semi-Major Axis (pkpk)", fontsize=20)
a1.set_xlabel("Cluster", fontsize=20)
a1.tick_params(labelsize=20)

a2 = sns.boxplot(ax=axs[1, 0], data=all_properties, x="Cluster", y="InitialMassWeightedStellarAge", showfliers=False, whis=1, palette=machine_palette, order=med_df["InitialMassWeightedStellarAge"].sort_values(ascending=False).index.to_list())
a2.set_ylabel("Stellar Age (Gyr)", fontsize=20)
a2.set_xlabel("Cluster", fontsize=20)
a2.tick_params(labelsize=20)

a3 = sns.boxplot(ax=axs[2, 0], data=all_properties, x="Cluster", y="StarFormationRate", showfliers=False, whis=1, palette=machine_palette, order=med_df["StarFormationRate"].sort_values(ascending=False).index.to_list())
a3.set_ylabel("Star Formation Rate (M$_{\odot}$yr$^{-1}$)", fontsize=20)
a3.set_xlabel("Cluster", fontsize=20)
a3.tick_params(labelsize=20)


b1 = sns.boxplot(ax=axs[0, 1], data=all_properties, x="Cluster", y=all_properties["MassType_Star"].div(1e10), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_Star"].sort_values(ascending=False).index.to_list())
b1.set_ylabel("Stellar Mass ($10^{10}$M$_{\odot}$)", fontsize=20)
b1.set_xlabel("Cluster", fontsize=20)
b1.tick_params(labelsize=20)
b1.set_yticks([0, 5, 10, 15])
b1.set_ylim(-0.1, 15.2)

b2 = sns.boxplot(ax=axs[1, 1], data=all_properties, x="Cluster", y=all_properties["MassType_DM"].div(1e12), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_Star"].sort_values(ascending=False).index.to_list())
b2.set_ylabel("Dark Matter Mass ($10^{12}$M$_{\odot}$)", fontsize=20)
b2.set_xlabel("Cluster", fontsize=20)
b2.tick_params(labelsize=20)

b3 = sns.boxplot(ax=axs[2, 1], data=all_properties, x="Cluster", y=all_properties["MassType_BH"].div(1e8), showfliers=False, whis=1, palette=machine_palette, order=med_df["MassType_Star"].sort_values(ascending=False).index.to_list())
b3.set_ylabel("Black Hole Mass ($10^{8}$M$_{\odot}$)", fontsize=20)
b3.set_xlabel("Cluster", fontsize=20)
b3.tick_params(labelsize=20)

disk_patch = mpatches.Patch(color=disk, label="Disk Structures")
bulge_patch = mpatches.Patch(color=bulge, label="Bulge Structures")
a1.legend(handles=[disk_patch, bulge_patch], bbox_to_anchor=(0.52, 0.93), loc='upper center', bbox_transform=fig.transFigure, ncol=2, prop={"size":20})

# plt.savefig("Plots/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution_all_features")
plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_physical_distribution", bbox_inches='tight')
plt.show()


# # all physical histogram
# fig, axs = plt.subplots(2, 3, figsize=(25, 10))
#
# bins = np.histogram_bin_edges(all_properties["re_r"], bins=20)
# combined_bins = np.sum(np.histogram(all_properties["re_r"], bins=bins[20:])[0])
# new_bins = np.concatenate((bins[:20], [bins[20]]))
# hist_counts = np.concatenate((np.histogram(all_properties["re_r"], bins=bins[:20])[0], [combined_bins]))
#
# a1 = sns.histplot(ax=axs[0, 0], data=all_properties, x="re_r", bins=new_bins, stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], legend=False)
# # a1 = sns.histplot(ax=axs[0], data=all_properties, x="re_r", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20, legend=False)
# a1.set_ylabel("Normalised Frequency", fontsize=20)
# a1.set_xlabel("Semi-Major Axis (pkpc)", fontsize=20)
# a1.set_yticks([])
# a1.tick_params(labelsize=20)
#
# a2 = sns.histplot(ax=axs[0, 1], data=all_properties, x="InitialMassWeightedStellarAge", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=20)
# a2.set_ylabel("Normalised Frequency", fontsize=20)
# a2.set_xlabel("Stellar Age (Gyr)", fontsize=20)
# a2.set_yticks([])
# a2.tick_params(labelsize=20)
# a2.legend(["More Featured Group", "Less Featured Group"], bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})
#
# a3 = sns.histplot(ax=axs[0, 2], data=all_properties, x="StarFormationRate", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=50, legend=False)
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
# a1 = sns.histplot(ax=axs[1, 0], data=all_properties, x=stellar_mass, bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
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
# a3 = sns.histplot(ax=axs[1, 2], data=all_properties, x=bh_mass, bins=new_bins, hue="Cluster", palette="colorblind", stat="probability", common_norm=False, hue_order=[1,0], legend=False)
# # a3 = sns.histplot(ax=axs[2], data=all_properties, x="MassType_BH", stat="probability", common_norm=False, hue="Cluster", palette="colorblind", hue_order=[1, 0], bins=150, legend=False)
# a3.set_ylabel("Normalised Frequency", fontsize=20)
# # a3.set_xlabel("Black Hole Mass ($10^{8} \mathrm{M}_{\odot}$)", fontsize=20)
# a3.set_xlabel("Black Hole Mass ($10^{8}$M$_{\odot}$)", fontsize=20)
# a3.set_yticks([])
# # a3.set_xlim(0, 0.5e9)
# a3.tick_params(labelsize=20)
#
# plt.savefig("Cluster Properties/" + str(encoding_dim) + "_feature_" + str(n_clusters) + "_cluster_all_physical_distribution", bbox_inches='tight')
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
total_elliptical = all_properties[all_properties["n_r"] > 2.5].shape[0]

print(str(total_spiral) + " Spirals (" + str(total_spiral/total) + ")")
print(str(total_elliptical) + " Ellipticals (" + str(total_elliptical/total) + ")")
print()


total_0 = all_properties[all_properties["Cluster"] == 0].shape[0]
total_0_spiral = all_properties[((all_properties["Cluster"] == 0) & (all_properties["n_r"] <= 2.5))].shape[0]
total_0_elliptical = all_properties[((all_properties["Cluster"] == 0) & (all_properties["n_r"] > 2.5))].shape[0]

print("Cluster 0 - " + str(total_0))
print("Cluster 0 - " + str(total_0_spiral) + " Spirals (" + str(total_0_spiral/total_0) + ")")
print("Cluster 0 - " + str(total_0_elliptical) + " Ellipticals (" + str(total_0_elliptical/total_0) + ")")
print()


total_1 = all_properties[all_properties["Cluster"] == 1].shape[0]
total_1_spiral = all_properties[((all_properties["Cluster"] == 1) & (all_properties["n_r"] <= 2.5))].shape[0]
total_1_elliptical = all_properties[((all_properties["Cluster"] == 1) & (all_properties["n_r"] > 2.5))].shape[0]

print("Cluster 1 - " + str(total_1))
print("Cluster 1 - " + str(total_1_spiral) + " Spirals (" + str(total_1_spiral/total_1) + ")")
print("Cluster 1 - " + str(total_1_elliptical) + " Ellipticals (" + str(total_1_elliptical/total_1) + ")")
print()


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




