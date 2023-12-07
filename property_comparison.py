import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from sklearn.cluster import KMeans
from matplotlib import image as mpimg


# extracted_features = np.flipud(np.rot90(extracted_features_original))


# set the encoding dimension (number of extracted features)
encoding_dim = 16



# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")

# perform clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')

# extract the clusters for each galaxy
clusters = kmeans.fit_predict(extracted_features)

centers = kmeans.cluster_centers_



print(clusters)
print(clusters.shape)
print(centers)
print(centers.shape)


# group_1 = df.loc[df['Cluster'] == 3, 'GalaxyID']


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

print(df1.shape)
print(df2.shape)
print(df.shape)

df["Cluster"] = clusters


group_1 = df.loc[df["Cluster"] == 0, "GalaxyID"].tolist()
group_2 = df.loc[df["Cluster"] == 1, "GalaxyID"].tolist()

print(np.array(group_1).shape)
print(np.array(group_2).shape)





fig, axs = plt.subplots(3, 6, figsize=(30,10))

for i in range(0, 3):

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1[i]) + ".png")
    axs[0, i].imshow(image)
    axs[0, i].get_xaxis().set_visible(False)
    axs[0, i].get_yaxis().set_visible(False)

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1[i+3]) + ".png")
    axs[1, i].imshow(image)
    axs[1, i].get_xaxis().set_visible(False)
    axs[1, i].get_yaxis().set_visible(False)

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1[i+6]) + ".png")
    axs[2, i].imshow(image)
    axs[2, i].get_xaxis().set_visible(False)
    axs[2, i].get_yaxis().set_visible(False)

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2[i]) + ".png")
    axs[0, i+3].imshow(image)
    axs[0, i+3].get_xaxis().set_visible(False)
    axs[0, i+3].get_yaxis().set_visible(False)

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2[i+3]) + ".png")
    axs[1, i+3].imshow(image)
    axs[1, i+3].get_xaxis().set_visible(False)
    axs[1, i+3].get_yaxis().set_visible(False)

    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2[i+6]) + ".png")
    axs[2, i+3].imshow(image)
    axs[2, i+3].get_xaxis().set_visible(False)
    axs[2, i+3].get_yaxis().set_visible(False)


axs[0,1].set_title("Group 1", pad=15)
axs[0,4].set_title("Group 2", pad=15)


plt.savefig("Plots/2_cluster_" + str(encoding_dim) + "_feature_originals")
plt.show()



# columns = []
# for i in range(1, encoding_dim+1):
#     columns.append("f" + str(i))
#
# extracted_feature_df = pd.DataFrame(extracted_features, columns=columns)
# extracted_feature_df["Category"] = clusters
#
#
# print(df)
#
# kws = dict(s=5, linewidth=0)
#
# sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")
#
# plt.savefig("Plots/2_cluster_" + str(encoding_dim) + "_features")
# plt.show()





# fig, axs = plt.subplots(6, 9, figsize=(25,20))
#
# for i in range(0, 9):
#
#     title = "Feature " + str(i+1)
#     axs[0, i].set_title(title)
#
#     # sns.scatterplot(ax=axs[0, i], x=extracted_features[i], y=ab_magnitude, hue=clusters, palette="colorblind")
#     # sns.scatterplot(ax=axs[1, i], x=extracted_features[i], y=mass, hue=clusters, palette="colorblind")
#     # sns.scatterplot(ax=axs[2, i], x=extracted_features[i], y=semi_major, hue=clusters, palette="colorblind")
#     # sns.scatterplot(ax=axs[3, i], x=extracted_features[i], y=sersic, hue=clusters, palette="colorblind")
#     # sns.scatterplot(ax=axs[4, i], x=extracted_features[i], y=axis_ratio, hue=clusters, palette="colorblind")
#     # sns.scatterplot(ax=axs[5, i], x=extracted_features[i], y=position_angle, hue=clusters, palette="colorblind")
#     #
#     # axs[0, i].get_legend().remove()
#     # axs[1, i].get_legend().remove()
#     # axs[2, i].get_legend().remove()
#     # axs[3, i].get_legend().remove()
#     # axs[4, i].get_legend().remove()
#     # axs[5, i].get_legend().remove()
#
#     sns.scatterplot(ax=axs[0, i], x=extracted_features[i], y=ab_magnitude)
#     sns.scatterplot(ax=axs[1, i], x=extracted_features[i], y=mass)
#     sns.scatterplot(ax=axs[2, i], x=extracted_features[i], y=semi_major)
#     sns.scatterplot(ax=axs[3, i], x=extracted_features[i], y=sersic)
#     sns.scatterplot(ax=axs[4, i], x=extracted_features[i], y=axis_ratio)
#     sns.scatterplot(ax=axs[5, i], x=extracted_features[i], y=position_angle)
#
#     # axs[0, i].scatter(extracted_features[i], ab_magnitude, s=2)
#     # axs[1, i].scatter(extracted_features[i], mass, s=2)
#     # axs[2, i].scatter(extracted_features[i], semi_major, s=2)
#     # axs[3, i].scatter(extracted_features[i], sersic, s=2)
#     # axs[4, i].scatter(extracted_features[i], axis_ratio, s=2)
#     # axs[5, i].scatter(extracted_features[i], position_angle, s=2)
#
# axs[0, 0].set_ylabel("AB Magnitude")
#
# axs[1, 0].set_ylabel("Stellar Mass")
# axs[2, 0].set_ylabel("Semi-Major Axis")
# axs[3, 0].set_ylabel("Sersic Index")
# axs[4, 0].set_ylabel("Axis Ratio")
# axs[5, 0].set_ylabel("Position Angle")
#
#
# plt.savefig("Plots/9_feature_property_comparison")
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
