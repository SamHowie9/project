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


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)




# set the encoding dimension (number of extracted features)
encoding_dim = 32

# set the number of clusters
n_clusters = 4


# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

# print(extracted_features)


# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")

# get hierarchical clusters
clusters = hierarchical.fit_predict(extracted_features)

# get hierarchical centers
clf = NearestCentroid()
clf.fit(extracted_features, clusters)
centers = clf.centroids_



# print(clusters)
# print(centers)








# load the two excel files into dataframes
df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
df2 = pd.read_csv("stab3510_supplemental_file/table2.csv", comment="#")


# account for hte validation data and remove final 200 elements
df1.drop(df1.tail(200).index, inplace=True)
df2.drop(df2.tail(200).index, inplace=True)



# extract relevant properties
galaxy_id = df1["GalaxyID"]
ab_magnitude = df1["galfit_mag"]
mass = df2["galfit_lmstar"]
semi_major = (df1["galfit_re"] + df2["galfit_re"]) / 2
sersic = (df1["galfit_n"] + df2["galfit_n"]) / 2
axis_ratio = (df1["galfit_q"] + df2["galfit_q"]) / 2
position_angle = (df1["galfit_PA"] + df2["galfit_PA"]) / 2


# create a new dataframe to contain all the relevant information about each galaxy
df = pd.DataFrame(columns=["GalaxyID", "galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA", "Cluster"])
df["GalaxyID"] = galaxy_id
df["galfit_mag"] = ab_magnitude
df["galfit_lmstar"] = mass
df["galfit_re"] = semi_major
df["galfit_n"] = sersic
df["galfit_q"] = axis_ratio
df["galfit_PA"] = position_angle
df["Cluster"] = clusters







# print(df[["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].mean())

# group_1 = df.loc[df["Cluster"] == 0, "GalaxyID"].tolist()
# group_2 = df.loc[df["Cluster"] == 1, "GalaxyID"].tolist()


group_1 = df.loc[df["Cluster"] == 0]
group_2 = df.loc[df["Cluster"] == 1]
group_3 = df.loc[df["Cluster"] == 2]
group_4 = df.loc[df["Cluster"] == 3]
group_5 = df.loc[df["Cluster"] == 4]
group_6 = df.loc[df["Cluster"] == 5]
group_7 = df.loc[df["Cluster"] == 6]

group_1_id = group_1["GalaxyID"]
group_2_id = group_2["GalaxyID"]
group_3_id = group_2["GalaxyID"]
group_4_id = group_2["GalaxyID"]

# print(group_1)
# print(group_2)

print(0, np.array(group_1).shape[0])
print(1, np.array(group_2).shape[0])
print(2, np.array(group_3).shape[0])
print(3, np.array(group_4).shape[0])
print(4, np.array(group_5).shape[0])
print(5, np.array(group_6).shape[0])
print(6, np.array(group_7).shape[0])






# mean_df = pd.DataFrame(columns=["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"])
#
# for i in range(0, n_clusters):
#     mean_cluster = df.loc[df["Cluster"] == i, ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].mean()
#     mean_df.loc[i] = mean_cluster
#
# mean_df["Cluster"] = list(range(0, n_clusters))
#
#
# g = sns.pairplot(data=mean_df, hue="Cluster", palette="colorblind", corner=True)
#
# cmap = sns.color_palette("cubehelix", as_cmap=True)
# sns.scatterplot(data=mean_df, x="galfit_mag", y="galfit_lmstar", hue="galfit_re", palette=cmap)
# sns.color_palette("cubehelix", as_cmap=True)
#
# plt.scatter(x=mean_df["galfit_lmstar"], y=mean_df["galfit_n"], s=100, linewidths=1, edgecolors="black", c=mean_df["galfit_mag"], cmap="plasma")
# plt.xlabel("Stellar Mass")
# plt.ylabel("Sersic Index")
#
#
# plt.colorbar(label="Absolute Magnitude")
#
#
# plt.savefig("Plots/stellar_mass_sersic_ab_" + str(n_clusters) + "_clusters")
# plt.show()





# # Apparent Magnitude, Stellar Mass, Semi-Major Axis, Sersic Index
# columns = ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]
#
# med_df = pd.DataFrame(columns=columns)
#
# for i in range(0, n_clusters):
#     med_cluster = df.loc[df["Cluster"] == i, ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].median()
#     med_df.loc[i] = med_cluster
#
# med_df["Cluster"] = list(range(0, n_clusters))
#
# print(med_df)
#
#
#
# centers_switch = np.flipud(np.rot90(centers))
#
# # print(pd.DataFrame(centers))
# # print(pd.DataFrame(centers_switch))
#
# print(centers_switch[14])
#
# # sns.scatterplot(x=centers_switch[14], y=med_df["galfit_PA"], hue=med_df["Cluster"])
# # plt.show()
#
#
# extracted_features_switch = np.flipud(np.rot90(extracted_features))
#
# # plt.scatter(x=centers, y=med_df["galfit_PA"])
# # plt.show()
#
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))
#
# bins=50
#
# # axs[0, 0].hist2d(x=extracted_features_switch[13], y=df["galfit_mag"], bins=(bins, bins), cmap=plt.cm.BuPu)
# # axs[0, 1].scatter(x=extracted_features_switch[13], y=df["galfit_mag"], s=2, alpha=0.1, c="black")
#
# # apparent magnitude vs f29
# sns.scatterplot(ax=axs[1, 0], x=extracted_features_switch[29], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# g1 = sns.scatterplot(ax=axs[1, 0], x=centers_switch[29], y=med_df["galfit_mag"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 29", ylabel="Apparent Magnitude", title="Apparent Magnitude vs Feature 29")
#
# # # apparent magnitude vs f14
# # sns.scatterplot(ax=axs[1, 0], x=extracted_features_switch[14], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# # g1 = sns.scatterplot(ax=axs[1, 0], x=centers_switch[14], y=med_df["galfit_mag"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# # g1.set(xlabel="Feature 14", ylabel="Apparent Magnitude", title="Apparent Magnitude")
#
# # # stellar mass vs f14
# # sns.scatterplot(ax=axs[0, 1], x=extracted_features_switch[14], y=df["galfit_lmstar"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# # g1 = sns.scatterplot(ax=axs[0, 1], x=centers_switch[14], y=med_df["galfit_lmstar"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# # g1.set(xlabel="Feature 14", ylabel="Stellar Mass", title="Stellar Mass")
#
# # # stellar mass vs f23
# # sns.scatterplot(ax=axs[1, 1], x=extracted_features_switch[23], y=df["galfit_lmstar"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# # g1 = sns.scatterplot(ax=axs[1, 1], x=centers_switch[23], y=med_df["galfit_lmstar"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# # g1.set(xlabel="Feature 23", ylabel="Stellar Mass", title="Stellar Mass")
#
# # semi-major axis vs f29
# sns.scatterplot(ax=axs[0, 0], x=extracted_features_switch[29], y=df["galfit_re"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# g1 = sns.scatterplot(ax=axs[0, 0], x=centers_switch[29], y=med_df["galfit_re"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 29", ylabel="Semi-Major Axis", title="Semi-Major Axis vs Feature 29", xlim=(5, 35), ylim=(0, 15))
#
# # # semi-major axis vs f25
# # sns.scatterplot(ax=axs[1, 2], x=extracted_features_switch[25], y=df["galfit_re"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# # g1 = sns.scatterplot(ax=axs[1, 2], x=centers_switch[25], y=med_df["galfit_re"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# # g1.set(xlabel="Feature 25", ylabel="Semi-Major Axis", title="Semi-Major Axis", xlim=(-10, 10), ylim=(0, 20))
#
# # sersic index vs f14
# sns.scatterplot(ax=axs[0, 1], x=extracted_features_switch[14], y=df["galfit_n"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# g1 = sns.scatterplot(ax=axs[0, 1], x=centers_switch[14], y=med_df["galfit_n"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=True)
# g1.set(xlabel="Feature 14", ylabel="Sersic Index", title="Sersic Index vs Feature 14", xlim=(-30, 10), ylim=(0, 5))
#
# # sns.kdeplot(ax=axs[1, 3], x=extracted_features_switch[14], y=df["galfit_n"], hue=df["Cluster"], palette="pastel", thresh=0.01, legend=False, fill=True, levels=2, alpha=0.5)
# # g2 = sns.scatterplot(ax=axs[1, 3], x=centers_switch[14], y=med_df["galfit_n"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# # g2.set(xlabel="Feature 14", ylabel="Sersic Index", title="Sersic Index", xlim=(-30, 10), ylim=(0, 5))
#
# # sersic index vs f23
# sns.scatterplot(ax=axs[1, 1], x=extracted_features_switch[23], y=df["galfit_n"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# g1 = sns.scatterplot(ax=axs[1, 1], x=centers_switch[23], y=med_df["galfit_n"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 23", ylabel="Sersic Index", title="Sersic Index vs Feature 23", xlim=(-30, 5), ylim=(0, 5))
#
#
# # move legend to cover whole plot
# sns.move_legend(axs[0, 1], "center left", bbox_to_anchor=(1.05, -0.1))
#
# # new_labels = ["Group 2-2", "Group 1-2", "Group 2-1", "Group 1-1"]
# new_labels = ["Group 2", "Group 1"]
#
# legend = axs[0, 1].get_legend()
# legend.set_title("Clusters")
# for current, new in zip(legend.texts, new_labels):
#     current.set_text(new)
#
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_clusters_" + str(encoding_dim) + "_features_property_correlation")
# plt.show()






# extracted_features_switch = np.flipud(np.rot90(extracted_features))
#
# correlation_df = pd.DataFrame(columns=["Apparent Magnitude", "Stellar Mass", "Semi-Major Axis", "Sersic Index", "Axis Ratio", "Position Angle"])
#
# for feature in range(0, len(extracted_features_switch)):
#
#     # create a list to contain the correlation between that feature and each property
#     correlation_list = []
#
#     # loop through each property
#     for gal_property in range(1, len(df.columns)-1):
#
#         # calculate the correlation between that extracted feature and that property
#         correlation = np.corrcoef(extracted_features_switch[feature], df.iloc[:, gal_property])[0][1]
#         correlation_list.append(correlation)
#
#     # add the correlation of that feature to the main dataframe
#     correlation_df.loc[len(correlation_df)] = correlation_list
#
# # set the figure size
# plt.figure(figsize=(12, 16))
#
# # sns.set(font_scale = 2)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=15)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
# ax.figure.axes[-1].yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(ax, 10)
#
#
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_property_correlation")
# plt.show()








# Apparent Magnitude, Stellar Mass, Semi-Major Axis, Sersic Index
columns = ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]

med_df = pd.DataFrame(columns=columns)

for i in range(0, n_clusters):
    med_cluster = df.loc[df["Cluster"] == i, ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].median()
    med_df.loc[i] = med_cluster

med_df["Cluster"] = list(range(0, n_clusters))

extracted_features_switch = np.flipud(np.rot90(extracted_features))
centers_switch = np.flipud(np.rot90(centers))

# create figure
# fig = plt.figure(constrained_layout=False, figsize=(15, 10))
#
# g1 = fig.add_gridspec(nrows=1, ncols=3, left=0.1, right=0.9, wspace=0.1, hspace=0.1, top=0.95, bottom=0.65)
# g2 = fig.add_gridspec(nrows=1, ncols=3, left=0.1, right=0.9, wspace=0.1, hspace=0.1, top=0.6, bottom=0.05)

# # plot probability distribution of each property for each cluster
# ax00 = fig.add_subplot(g1[0])
# h0 = sns.histplot(ax=ax00, data=df, x="galfit_mag", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h0.set(xlabel=None, ylabel=None, title="Apparent Magnitude")
#
# ax01 = fig.add_subplot(g1[1])
# h1 = sns.histplot(ax=ax01, data=df, x="galfit_lmstar", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h1.set(xlabel=None, ylabel=None, title="Stellar Mass")
#
# ax02 = fig.add_subplot(g1[2])
# h2 = sns.histplot(ax=ax02, data=df, x="galfit_re", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly", binrange=(0, 25))
# h2.set(xlabel=None, ylabel=None, title="Semi-Major Axis")
#
# ax03 = fig.add_subplot(g1[3])
# h3 = sns.histplot(ax=ax03, data=df, x="galfit_n", hue="Cluster", palette="bright", linewidth=0, legend=True, element="poly")
# h3.set(xlabel=None, ylabel=None, title="Sersic Index")


# # apparent magnitude vs f29
# ax00 = fig.add_subplot(g1[0])
# sns.scatterplot(ax=ax00, x=extracted_features_switch[29], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# ax000 = fig.add_subplot(g1[0], sharex=ax00, sharey=ax00)
# g1 = sns.scatterplot(ax=ax000, x=centers_switch[29], y=med_df["galfit_mag"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 29", ylabel="Apparent Magnitude", title="Apparent Magnitude vs Feature 29")
#
#
# # semi-major axis vs f29
# ax01 = fig.add_subplot(g1[1])
# sns.scatterplot(ax=ax01, x=extracted_features_switch[29], y=df["galfit_re"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# ax011 = fig.add_subplot(g1[1], sharex=ax01, sharey=ax01)
# g1 = sns.scatterplot(ax=ax011, x=centers_switch[29], y=med_df["galfit_re"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False)
# g1.set(xlabel="Feature 29", ylabel="Semi-Major Axis", title="Semi-Major Axis vs Feature 29", xlim=(5, 35), ylim=(0, 15))
#
# # sersic index vs f14
# ax02 = fig.add_subplot(g1[2])
# sns.scatterplot(ax=ax02, x=extracted_features_switch[14], y=df["galfit_n"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
# # g1 = sns.scatterplot(ax=ax02, x=centers_switch[14], y=med_df["galfit_n"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=True)
# # g1.set(xlabel="Feature 14", ylabel="Sersic Index", title="Sersic Index vs Feature 14", xlim=(-30, 10), ylim=(0, 5))
#
#
#
#
#
# ax10 = fig.add_subplot(g2[0])
# b0 = sns.boxplot(ax=ax10, data=df, x="Cluster", y="galfit_mag", showfliers=False, whis=1, palette="pastel")
# b0.set(xlabel=None, ylabel=None, xticklabels=["Group 2", "Group 1"])
#
# # ax11 = fig.add_subplot(g2[1])
# # b1 = sns.boxplot(ax=ax11, data=df, x="Cluster", y="galfit_lmstar", showfliers=False, whis=1, palette="pastel")
# # b1.set(xlabel=None, ylabel=None, xticklabels=["Group 2", "Group 1"])
#
# ax12 = fig.add_subplot(g2[1])
# b2 = sns.boxplot(ax=ax12, data=df, x="Cluster", y="galfit_re", showfliers=False, whis=1, palette="pastel")
# b2.set(xlabel=None, ylabel=None, xticklabels=["Group 2", "Group 1"])
#
# ax13 = fig.add_subplot(g2[2])
# b3 = sns.boxplot(ax=ax13, data=df, x="Cluster", y="galfit_n", showfliers=False, whis=1, palette="pastel")
# b3.set(xlabel=None, ylabel=None, xticklabels=["Group 2", "Group 1"])


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# apparent magnitude vs f29
# sns.scatterplot(ax=axs[0, 0], x=extracted_features_switch[29], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
sns.scatterplot(ax=axs[0, 0], x=extracted_features_switch[29], y=df["galfit_mag"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False, hue_order=[3, 1, 2, 0])
g1 = sns.scatterplot(ax=axs[0, 0], x=centers_switch[29], y=med_df["galfit_mag"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False, hue_order=[3, 1, 2, 0])
g1.set(xlabel="Feature 29", ylabel="Apparent Magnitude", title="Apparent Magnitude vs Feature 29")

# semi-major axis vs f29
# sns.scatterplot(ax=axs[0, 1], x=extracted_features_switch[29], y=df["galfit_re"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
sns.scatterplot(ax=axs[0, 1], x=extracted_features_switch[29], y=df["galfit_re"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False, hue_order=[3, 1, 2, 0])
g1 = sns.scatterplot(ax=axs[0, 1], x=centers_switch[29], y=med_df["galfit_re"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=False, hue_order=[3, 1, 2, 0])
g1.set(xlabel="Feature 29", ylabel="Semi-Major Axis", title="Semi-Major Axis vs Feature 29", xlim=(5, 35), ylim=(0, 15))

# sersic index vs f14
# sns.scatterplot(ax=axs[0, 2], x=extracted_features_switch[14], y=df["galfit_n"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False)
sns.scatterplot(ax=axs[0, 2], x=extracted_features_switch[14], y=df["galfit_n"], alpha=0.05, linewidth=0, hue=df["Cluster"], palette="pastel", legend=False, hue_order=[3, 1, 2, 0])
g1 = sns.scatterplot(ax=axs[0, 2], x=centers_switch[14], y=med_df["galfit_n"], s=100, hue=med_df["Cluster"], palette="colorblind", legend=True, hue_order=[3, 1, 2, 0])
g1.set(xlabel="Feature 14", ylabel="Sersic Index", title="Sersic Index vs Feature 14", xlim=(-30, 10), ylim=(0, 5))

# h0 = sns.histplot(ax=axs[0, 0], data=df, x="galfit_mag", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h0.set(xlabel=None, ylabel=None, title="Apparent Magnitude")
# h1 = sns.histplot(ax=axs[0, 1], data=df, x="galfit_lmstar", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h1.set(xlabel=None, ylabel=None, title="Stellar Mass")
# h2 = sns.histplot(ax=axs[0, 2], data=df, x="galfit_re", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly", binrange=(0, 25))
# h2.set(xlabel=None, ylabel=None, title="Semi-Major Axis")
# h3 = sns.histplot(ax=axs[0, 3], data=df, x="galfit_n", hue="Cluster", palette="bright", linewidth=0, legend=True, element="poly")
# h3.set(xlabel=None, ylabel=None, title="Sersic Index")

# plot mean value of each property for each cluster
b0 = sns.boxplot(ax=axs[1, 0], data=df, x="Cluster", y="galfit_mag", showfliers=False, whis=1, palette="colorblind", order=[3, 1, 2, 0])
# b0.set(xlabel=None, ylabel=None, title="Apparent Magnitude", xticklabels=["Group 2", "Group 1"])
b0.set(xlabel=None, ylabel=None, title="Apparent Magnitude", xticklabels=["Group 1-1", "Group 1-2", "Group 2-1", "Group 2-2"])
b2 = sns.boxplot(ax=axs[1, 1], data=df, x="Cluster", y="galfit_re", showfliers=False, whis=1, palette="colorblind", order=[3, 1, 2, 0])
# b2.set(xlabel=None, ylabel=None, title="Semi-Major Axis", xticklabels=["Group 2", "Group 1"])
b2.set(xlabel=None, ylabel=None, title="Semi-Major Axis", xticklabels=["Group 1-1", "Group 1-2", "Group 2-1", "Group 2-2"])
b3 = sns.boxplot(ax=axs[1, 2], data=df, x="Cluster", y="galfit_n", showfliers=False, whis=1, palette="colorblind", alpha=0.5, order=[3, 1, 2, 0])
# b3.set(xlabel=None, ylabel=None, title="Sersic Index", xticklabels=["Group 2", "Group 1"])
b3.set(xlabel=None, ylabel=None, title="Sersic Index", xticklabels=["Group 1-1", "Group 1-2", "Group 2-1", "Group 2-2"])

sns.move_legend(axs[0, 2], "center left", bbox_to_anchor=(1.05, 0.5))

# new_labels = ["Group 2", "Group 1"]
new_labels = ["Group 1-1", "Group 1-2", "Group 2-1", "Group 2-2"]

legend = axs[0, 2].get_legend()
legend.set_title("Clusters")
for current, new in zip(legend.texts, new_labels):
    current.set_text(new)


plt.savefig("Plots/" + str(n_clusters) + "_cluster_properties.png")
plt.show()










# columns = []
# for i in range(1, encoding_dim+1):
#     columns.append("f" + str(i))
#
# extracted_feature_df = pd.DataFrame(extracted_features, columns=columns)
# extracted_feature_df["Cluster"] = clusters
#
#
# print(extracted_feature_df)
#
# kws = dict(s=5, linewidth=0)
#
# sns.pairplot(extracted_feature_df, corner=True, hue="Cluster", plot_kws=kws, palette="colorblind")
#
# plt.savefig("Plots/2_cluster_" + str(encoding_dim) + "_features")
# plt.show()








# extracted_features = np.flipud(np.rot90(extracted_features))
#
# fig, axs = plt.subplots(encoding_dim, 6, figsize=(20,60))
#
# for i in range(0, encoding_dim):
#
#     bins = 75
#
#     axs[i, 0].hist2d(y=extracted_features[i], x=ab_magnitude, bins=(bins, bins), cmap=plt.cm.BuPu)
#     axs[i, 1].hist2d(y=extracted_features[i], x=mass, bins=(bins, bins), cmap=plt.cm.BuPu)
#     axs[i, 2].hist2d(y=extracted_features[i], x=semi_major, bins=(bins, bins), cmap=plt.cm.BuPu)
#     axs[i, 3].hist2d(y=extracted_features[i], x=sersic, bins=(bins, bins), cmap=plt.cm.BuPu)
#     axs[i, 4].hist2d(y=extracted_features[i], x=axis_ratio, bins=(bins, bins), cmap=plt.cm.BuPu)
#     axs[i, 5].hist2d(y=extracted_features[i], x=position_angle, bins=(bins, bins), cmap=plt.cm.BuPu)
#
#
# axs[0, 0].set_title("AB Magnitude")
# axs[0, 1].set_title("Stellar Mass")
# axs[0, 2].set_title("Semi-Major Axis")
# axs[0, 3].set_title("Sersic Index")
# axs[0, 4].set_title("Axis Ratio")
# axs[0, 5].set_title("Position Angle")
#
#
# plt.yticks(rotation=90)
#
# plt.savefig("Plots/" + str(encoding_dim) + "_feature_property_comparison")
# plt.show()






# df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"])
#
#
# # create the pairplot with custom marker size
# kws = dict(s=10)
# g = sns.pairplot(df, corner=True, plot_kws=kws)
#
#
# # function to add the correlation coefficient to the plots
# def corrfunc(x, y, ax=None, color=None):
#     # find the correlation coefficient and round to 3 dp
#     correlation = np.corrcoef(x, y)[0][1]
#     correlation = np.round(correlation, decimals=3)
#
#     # annotate the plot with the correlation coefficient
#     ax = ax or plt.gca()
#     ax.annotate(("ρ = " + str(correlation)), xy=(0.1, 1), xycoords=ax.transAxes)
#
#
# # add the correlation coefficient to each of the scatter plots
# g.map_lower(corrfunc)
#
# # add some vertical space between the plots (given we are adding the correlation coefficients
# plt.subplots_adjust(hspace=0.2)
#
#
# plt.savefig("Plots/9_feature_histogram_new")
# plt.show()
