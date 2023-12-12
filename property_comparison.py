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





# set the encoding dimension (number of extracted features)
encoding_dim = 32

# set the number of clusters
n_clusters = 2


# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

# print(extracted_features)



# perform hierarchical ward clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")

# get hierarchical clusters
clusters_h = hierarchical.fit_predict(extracted_features)

# get hierarchical centers
clf = NearestCentroid()
clf.fit(extracted_features, clusters_h)
centers_h = clf.centroids_


clusters = clusters_h
centers = centers_h

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

group_1_id = group_1["GalaxyID"]
group_2_id = group_2["GalaxyID"]
group_3_id = group_2["GalaxyID"]
group_4_id = group_2["GalaxyID"]

# print(group_1)
# print(group_2)

print(np.array(group_1).shape)
print(np.array(group_2).shape)
print(np.array(group_3).shape)
print(np.array(group_4).shape)

print(np.array(group_1).shape[0])
print(np.array(group_2).shape[0])
print(np.array(group_3).shape[0])
print(np.array(group_4).shape[0])





# mean_df = pd.DataFrame(columns=["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"])
#
# for i in range(0, n_clusters):
#     mean_cluster = df.loc[df["Cluster"] == i, ["galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA"]].mean()
#     mean_df.loc[i] = mean_cluster
#
# mean_df["Cluster"] = list(range(0, n_clusters))


# g = sns.pairplot(data=mean_df, hue="Cluster", palette="colorblind", corner=True)

# cmap = sns.color_palette("cubehelix", as_cmap=True)
# sns.scatterplot(data=mean_df, x="galfit_mag", y="galfit_lmstar", hue="galfit_re", palette=cmap)
# sns.color_palette("cubehelix", as_cmap=True)

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











# fig, axs = plt.subplots(2, 6, figsize=(25, 8))
#
# h0 = sns.histplot(ax=axs[0, 0], data=df, x="galfit_mag", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h0.set(xlabel=None, ylabel=None, title="Absolute Magnitude")
# h1 = sns.histplot(ax=axs[0, 1], data=df, x="galfit_lmstar", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h1.set(xlabel=None, ylabel=None, title="Stellar Mass")
# h2 = sns.histplot(ax=axs[0, 2], data=df, x="galfit_re", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h2.set(xlabel=None, ylabel=None, title="Semi-Major Axis")
# h3 = sns.histplot(ax=axs[0, 3], data=df, x="galfit_n", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h3.set(xlabel=None, ylabel=None, title="Sersic Index")
# h4 = sns.histplot(ax=axs[0, 4], data=df, x="galfit_q", hue="Cluster", palette="bright", linewidth=0, legend=False, element="poly")
# h4.set(xlabel=None, ylabel=None, title="Axis Ratio")
# h5 = sns.histplot(ax=axs[0, 5], data=df, x="galfit_PA", hue="Cluster", palette="bright", linewidth=0, legend=True, element="poly")
# h5.set(xlabel=None, ylabel=None, title="Position Angle")
#
# b0 = sns.boxplot(ax=axs[1, 0], data=df, x="Cluster", y="galfit_mag", showfliers=False, whis=0, palette="pastel")
# b0.set(xlabel=None, ylabel=None, xticklabels=[])
# b1 = sns.boxplot(ax=axs[1, 1], data=df, x="Cluster", y="galfit_lmstar", showfliers=False, whis=0, palette="pastel")
# b1.set(xlabel=None, ylabel=None, xticklabels=[])
# b2 = sns.boxplot(ax=axs[1, 2], data=df, x="Cluster", y="galfit_re", showfliers=False, whis=0, palette="pastel")
# b2.set(xlabel=None, ylabel=None, xticklabels=[])
# b3 = sns.boxplot(ax=axs[1, 3], data=df, x="Cluster", y="galfit_n", showfliers=False, whis=0, palette="pastel")
# b3.set(xlabel=None, ylabel=None, xticklabels=[])
# b4 = sns.boxplot(ax=axs[1, 4], data=df, x="Cluster", y="galfit_q", showfliers=False, whis=0, palette="pastel")
# b4.set(xlabel=None, ylabel=None, xticklabels=[])
# b5 = sns.boxplot(ax=axs[1, 5], data=df, x="Cluster", y="galfit_PA", showfliers=False, whis=0, palette="pastel")
# b5.set(xlabel=None, ylabel=None, xticklabels=[])
#
# sns.move_legend(axs[0, 5], "center left", bbox_to_anchor=(1.05, -0.1))
#
# labels = []
# for i in range(1, n_clusters+1):
#     labels.append("Group " + str(i))
#
# legend = axs[0, 5].get_legend()
# legend.set_title("Clusters")
# for current, new in zip(legend.texts, labels):
#     current.set_text(new)
#
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_properties.png")
# plt.show()








group_1_random_index = random.sample(range(0, len(group_1_id)), 9)
group_2_random_index = random.sample(range(0, len(group_2_id)), 9)
group_3_random_index = random.sample(range(0, len(group_3_id)), 9)
group_4_random_index = random.sample(range(0, len(group_4_id)), 9)

group_1_random = group_1_id.iloc[group_1_random_index].tolist()
group_2_random = group_2_id.iloc[group_2_random_index].tolist()
group_3_random = group_3_id.iloc[group_3_random_index].tolist()
group_4_random = group_4_id.iloc[group_4_random_index].tolist()

# group_1_random = [3518865, 3533021, 10108400, 10452290, 9195988, 10625818, 17097594, 7164803, 9563813]
# group_2_random = [12485051, 9599139, 14266206, 9032934, 17858355, 10372952, 15996483, 9542933, 7144268]

print(group_1_random)
print(group_2_random)
print(group_3_random)
print(group_4_random)

# fig, axs = plt.subplots(3, 6, figsize=(20,10))


fig = plt.figure(constrained_layout=False, figsize=(20, 10))

gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05)
gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05)
# gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05, top=0.95, bottom=0.55)
# gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05, top=0.95, bottom=0.55)
# gs3 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05, top=0.45, bottom=0.05)
# gs4 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05, top=0.45, bottom=0.05)

for i in range(0, 3):

    g1_ax1 = fig.add_subplot(gs1[0, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[i]) + ".png")
    # g1_ax1.imshow(image)
    g1_ax1.get_xaxis().set_visible(False)
    g1_ax1.get_yaxis().set_visible(False)

    g1_ax2 = fig.add_subplot(gs1[1, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[i+3]) + ".png")
    # g1_ax2.imshow(image)
    g1_ax2.get_xaxis().set_visible(False)
    g1_ax2.get_yaxis().set_visible(False)

    g1_ax3 = fig.add_subplot(gs1[2, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[i+6]) + ".png")
    # g1_ax3.imshow(image)
    g1_ax3.get_xaxis().set_visible(False)
    g1_ax3.get_yaxis().set_visible(False)


    g2_ax1 = fig.add_subplot(gs2[0, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2_random[i]) + ".png")
    # g2_ax1.imshow(image)
    g2_ax1.get_xaxis().set_visible(False)
    g2_ax1.get_yaxis().set_visible(False)

    g2_ax2 = fig.add_subplot(gs2[1, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2_random[i+3]) + ".png")
    # g2_ax2.imshow(image)
    g2_ax2.get_xaxis().set_visible(False)
    g2_ax2.get_yaxis().set_visible(False)

    g2_ax3 = fig.add_subplot(gs2[2, i])
    # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2_random[i+6]) + ".png")
    # g2_ax3.imshow(image)
    g2_ax3.get_xaxis().set_visible(False)
    g2_ax3.get_yaxis().set_visible(False)


    # g3_ax1 = fig.add_subplot(gs3[0, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_3_random[i]) + ".png")
    # # g3_ax1.imshow(image)
    # g3_ax1.get_xaxis().set_visible(False)
    # g3_ax1.get_yaxis().set_visible(False)
    # #
    # g3_ax2 = fig.add_subplot(gs3[1, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_3_random[i+3]) + ".png")
    # # g3_ax2.imshow(image)
    # g3_ax2.get_xaxis().set_visible(False)
    # g3_ax2.get_yaxis().set_visible(False)
    # #
    # g3_ax3 = fig.add_subplot(gs3[2, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_3_random[i+6]) + ".png")
    # # g3_ax3.imshow(image)
    # g3_ax3.get_xaxis().set_visible(False)
    # g3_ax3.get_yaxis().set_visible(False)
    # #
    # #
    # g4_ax1 = fig.add_subplot(gs4[0, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_4_random[i]) + ".png")
    # # g4_ax1.imshow(image)
    # g4_ax1.get_xaxis().set_visible(False)
    # g4_ax1.get_yaxis().set_visible(False)
    # #
    # g4_ax2 = fig.add_subplot(gs4[1, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_4_random[i+3]) + ".png")
    # # g4_ax2.imshow(image)
    # g4_ax2.get_xaxis().set_visible(False)
    # g4_ax2.get_yaxis().set_visible(False)
    # #
    # g4_ax3 = fig.add_subplot(gs4[2, i])
    # # image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_4_random[i+6]) + ".png")
    # # g4_ax3.imshow(image)
    # g4_ax3.get_xaxis().set_visible(False)
    # g4_ax3.get_yaxis().set_visible(False)




    if i == 1:
        g1_ax1.set_title(("Group 1 (" + str(np.array(group_1).shape[0]) + ")"), fontsize=25, pad=20)
        g2_ax1.set_title(("Group 2 (" + str(np.array(group_2).shape[0]) + ")"), fontsize=25, pad=20)
        # g3_ax1.set_title(("Group 3 (" + str(np.array(group_3).shape[0]) + ")"), fontsize=25, pad=20)
        # g4_ax1.set_title(("Group 4 (" + str(np.array(group_4).shape[0]) + ")"), fontsize=25, pad=20)



plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(encoding_dim) + "_feature_originals_4")
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
