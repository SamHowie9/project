import eagleSqlTools as sql
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import textwrap

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)




# set the encoding dimension (number of extracted features)
encoding_dim = 32

# set the number of clusters
n_clusters = 1



# sersic >= 4
# use median for center
# check how galfit properties are amde
# resise images, use absolute magnitude distrbution
# look at mean redshift, look at redshift of images (trayford paper), 0.1
#
# use abosulte mag to find sweet spot for cropping or resizing


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

# load the two excel files into dataframes
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
df.drop(df.tail(200).index, inplace=True)

# print(df["GalaxyID"].tolist())

# load the data
df_extra = pd.read_csv("RefL0100N1504_Subhalo.csv", comment="#")


df = df.merge(df_extra, how="left", on="GalaxyID")
df = df.dropna()

df = df[["BlackHoleMass", "BlackHoleMassAccretionRate", "InitialMassWeightedBirthZ", "InitialMassWeightedStellarAge", "MassType_Star", "MassType_DM", "MassType_Gas", "MassType_BH", "StarFormationRate", "StellarInitialMass"]]
df["Cluster"] = clusters



extracted_features_switch = np.flipud(np.rot90(extracted_features))
centers_switch = np.flipud(np.rot90(centers))


mean_df = pd.DataFrame(columns=["BlackHoleMass", "BlackHoleMassAccretionRate", "InitialMassWeightedBirthZ", "InitialMassWeightedStellarAge", "MassType_Star", "MassType_DM", "MassType_Gas", "MassType_BH", "StarFormationRate", "StellarInitialMass"])


for i in range(0, n_clusters):
    mean_cluster = df.loc[df["Cluster"] == i, ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].mean()
    mean_df.loc[i] = mean_cluster

mean_df["Cluster"] = list(range(0, n_clusters))

print(mean_df)


# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
# # absolute magnitude vs f15
# sns.scatterplot(ax=axs[0, 0], x=extracted_features_switch[13], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# g1 = sns.scatterplot(ax=axs[0, 0], x=centers_switch[13], y=mean_df["galfit_mag"], s=100, hue=mean_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 13", ylabel="Absolute Magnitude", title="Absolute Magnitude")
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_clusters_" + str(encoding_dim) + "_features_extra_property_correlation")
# plt.show()






# correlation_df = pd.DataFrame(columns=["Black Hole Mass", "Black Hole Accretion Rate", "Mean Redshift of Star Formation", "Mean Age of Stars", "Stellar Mass", "Dark Matter Mass", "Gas Mass", "Black Hole Mass", "Star Formation Rate", "Initial Stellar Mass"])
# # correlation_df = pd.DataFrame(columns=["Redshift", "BlackHoleMass", "BlackHoleMassAccretionRate", "InitialMassWeightedBirthZ", "InitialMassWeightedStellarAge", "MassType_Star", "MassType_DM", "MassType_Gas", "MassType_BH", "StarFormationRate", "StellarInitialMass"])
#
# for feature in range(0, len(extracted_features_switch)):
#
#     # create a list to contain the correlation between that feature and each property
#     correlation_list = []
#
#     # loop through each property
#     for gal_property in range(0, len(df.columns)-1):
#
#         # calculate the correlation between that extracted feature and that property
#         correlation = np.corrcoef(extracted_features_switch[feature], df.iloc[:, gal_property])[0][1]
#         correlation_list.append(correlation)
#
#     print(correlation_list)
#     # add the correlation of that feature to the main dataframe
#     correlation_df.loc[len(correlation_df)] = correlation_list
#
# # set the figure size
# plt.figure(figsize=(16, 16))
#
# # sns.set(font_scale = 2)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=10)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
# ax.figure.axes[-1].yaxis.label.set_size(10)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=10)
#
#
# wrap_labels(ax, 10)
#
#
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_extra_property_correlation")
# plt.show()