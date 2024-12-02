import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random




# set the encoding dimension (number of extracted features)
encoding_dim = 38

# set the number of clusters
n_clusters = 14


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")




# # load the extracted features
# extracted_features = np.load("Features/" + str(encoding_dim) + "_features_3.npy")
#
#
# # perform hierarchical ward clustering
# hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
#
# # get hierarchical clusters
# clusters = hierarchical.fit_predict(extracted_features)
#
# all_properties["Cluster"] = clusters










# create the figure for the plot
fig = plt.figure(constrained_layout=False, figsize=(20, 10))

# create the subfigures for the plot (each group)
gs1 = fig.add_gridspec(nrows=4, ncols=4, left=0.05, right=0.49, wspace=0.05, hspace=0.05)
gs2 = fig.add_gridspec(nrows=4, ncols=4, left=0.51, right=0.95, wspace=0.05, hspace=0.05)



# vae and pca
galaxies_1 = [9354175, 8649269, 12008778, 10044250, 8097697, 3528200, 8903544, 9827336, 10148850, 8121522, 14515322, 13207800, 9216031, 10390334, 10669399, 8643938]
galaxies_2 = [10835614, 12184457, 9674774, 8439349, 10078536, 17079994, 8086783, 9026380, 12192715, 12115375, 16750450, 64010, 1032412, 8585110, 11533908, 18320344]

# vae and pca


count = 0

for i in range(0, 3):
    for j in range(0, 3):

        g1_ax = fig.add_subplot(gs1[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxies_1[count]) + ".png")
        g1_ax.imshow(image)
        g1_ax.get_xaxis().set_visible(False)
        g1_ax.get_yaxis().set_visible(False)

        g2_ax = fig.add_subplot(gs2[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxies_2[count]) + ".png")
        g2_ax.imshow(image)
        g2_ax.get_xaxis().set_visible(False)
        g2_ax.get_yaxis().set_visible(False)

        # set group title for middle plot of each group
        if i == 0 and j == 1:

            # g1_ax.set_title("More Featured (Spiral-Like)", fontsize=25, pad=20)
            # g2_ax.set_title("Less Featured (Elliptical-Like)", fontsize=25, pad=20)

            g1_ax.set_title("Less Featured (Elliptical-Like)", fontsize=25, pad=20)
            g2_ax.set_title("More Featured (Spiral-Like)", fontsize=25, pad=20)

        count += 1



plt.savefig("Variational Eagle/Cluster Plots/vae_2_cluster_" + str(encoding_dim) + "_feature_originals")
plt.show()






# # separate the dataframe into groups based on their cluster
# group_1 = df.loc[df["Cluster"] == 0]
# group_2 = df.loc[df["Cluster"] == 1]
# group_3 = df.loc[df["Cluster"] == 2]
# group_4 = df.loc[df["Cluster"] == 3]
#
# # get a list of 9 random indices for each group
# group_1_random_index = random.sample(range(0, len(group_1)), 9)
# group_2_random_index = random.sample(range(0, len(group_2)), 9)
# group_3_random_index = random.sample(range(0, len(group_3)), 9)
# group_4_random_index = random.sample(range(0, len(group_4)), 9)
#
# # get the galaxy id of each of the random indices for each group
# group_1_random = group_1["GalaxyID"].iloc[group_1_random_index].tolist()
# group_2_random = group_2["GalaxyID"].iloc[group_2_random_index].tolist()
# group_3_random = group_3["GalaxyID"].iloc[group_3_random_index].tolist()
# group_4_random = group_4["GalaxyID"].iloc[group_4_random_index].tolist()
#
# print(group_1_random)
# print(group_2_random)
# print(group_3_random)
# print(group_4_random)
#
# print(group_1.shape[0])
# print(group_2.shape[0])
# print(group_3.shape[0])
# print(group_4.shape[0])
#
# # create figure
# fig = plt.figure(constrained_layout=False, figsize=(20, 20))
#
# # create sub figures within main figure, specify their location
# gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05, top=0.95, bottom=0.55)
# gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05, top=0.95, bottom=0.55)
# gs3 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05, top=0.45, bottom=0.05)
# gs4 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05, top=0.45, bottom=0.05)
#
# count = 0
#
# for i in range(0, 3):
#     for j in range(0, 3):
#
#         g1_ax = fig.add_subplot(gs1[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_4_random[count]) + ".png")
#         g1_ax.imshow(resize_image(image, 0.06))
#         g1_ax.get_xaxis().set_visible(False)
#         g1_ax.get_yaxis().set_visible(False)
#
#         g2_ax = fig.add_subplot(gs2[i, j])
#         print(".................")
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_3_random[count]) + ".png")
#         g2_ax.imshow(resize_image(image, 0.06))
#         g2_ax.get_xaxis().set_visible(False)
#         g2_ax.get_yaxis().set_visible(False)
#         print("................")
#
#         g3_ax = fig.add_subplot(gs3[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2_random[count]) + ".png")
#         g3_ax.imshow(resize_image(image, 0.06))
#         g3_ax.get_xaxis().set_visible(False)
#         g3_ax.get_yaxis().set_visible(False)
#
#         g4_ax = fig.add_subplot(gs4[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[count]) + ".png")
#         g4_ax.imshow(resize_image(image, 0.06))
#         g4_ax.get_xaxis().set_visible(False)
#         g4_ax.get_yaxis().set_visible(False)
#
#         # set group title for middle plot of each group
#         if i == 0 and j == 1:
#             g1_ax.set_title(("Group 1-1 (" + str(np.array(group_4).shape[0]) + ")"), fontsize=30, pad=20)
#             g2_ax.set_title(("Group 2-1 (" + str(np.array(group_3).shape[0]) + ")"), fontsize=30, pad=20)
#             g3_ax.set_title(("Group 1-2 (" + str(np.array(group_2).shape[0]) + ")"), fontsize=30, pad=20)
#             g4_ax.set_title(("Group 2-2 (" + str(np.array(group_1).shape[0]) + ")"), fontsize=30, pad=20)
#
#         count += 1







# # separate the dataframe into groups based on their cluster
# group_1 = df.loc[df["Cluster"] == 0]
# group_2 = df.loc[df["Cluster"] == 1]
# group_3 = df.loc[df["Cluster"] == 2]
# group_4 = df.loc[df["Cluster"] == 3]
# group_5 = df.loc[df["Cluster"] == 4]
# group_6 = df.loc[df["Cluster"] == 5]
# group_7 = df.loc[df["Cluster"] == 6]
#
# # get a list of 9 random indices for each group
# group_1_random_index = random.sample(range(0, len(group_1)), 9)
# group_2_random_index = random.sample(range(0, len(group_2)), 9)
# group_3_random_index = random.sample(range(0, len(group_3)), 9)
# group_4_random_index = random.sample(range(0, len(group_4)), 9)
# group_5_random_index = random.sample(range(0, len(group_5)), 9)
# group_6_random_index = random.sample(range(0, len(group_6)), 9)
# group_7_random_index = random.sample(range(0, len(group_7)), 9)
#
# # get the galaxy id of each of the random indices for each group
# group_1_random = group_1["GalaxyID"].iloc[group_1_random_index].tolist()
# group_2_random = group_2["GalaxyID"].iloc[group_2_random_index].tolist()
# group_3_random = group_3["GalaxyID"].iloc[group_3_random_index].tolist()
# group_4_random = group_4["GalaxyID"].iloc[group_4_random_index].tolist()
# group_5_random = group_5["GalaxyID"].iloc[group_5_random_index].tolist()
# group_6_random = group_6["GalaxyID"].iloc[group_6_random_index].tolist()
# group_7_random = group_7["GalaxyID"].iloc[group_7_random_index].tolist()
#
# print(group_1_random)
# print(group_2_random)
# print(group_3_random)
# print(group_4_random)
# print(group_5_random)
# print(group_6_random)
# print(group_7_random)
#
# # create figure
# fig = plt.figure(constrained_layout=False, figsize=(20, 10))
#
# # create sub figures within main figure, specify their location
# gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.225, wspace=0.05, hspace=0.05, top=0.85, bottom=0.55)
# gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.275, right=0.45, wspace=0.05, hspace=0.05, top=0.85, bottom=0.55)
# gs3 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.725, wspace=0.05, hspace=0.05, top=0.85, bottom=0.55)
# gs4 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.225, wspace=0.05, hspace=0.05, top=0.4, bottom=0.1)
# gs5 = fig.add_gridspec(nrows=3, ncols=3, left=0.275, right=0.45, wspace=0.05, hspace=0.05, top=0.4, bottom=0.1)
# gs6 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.725, wspace=0.05, hspace=0.05, top=0.4, bottom=0.1)
# gs7 = fig.add_gridspec(nrows=3, ncols=3, left=0.775, right=0.95, wspace=0.05, hspace=0.05, top=0.4, bottom=0.1)
#
# # gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.25, wspace=0.05, hspace=0.05, top=0.9, bottom=0.525)
# # gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.25, right=0.45, wspace=0.05, hspace=0.05, top=0.9, bottom=0.525)
# # gs3 = fig.add_gridspec(nrows=3, ncols=3, left=0.6, right=0.8, wspace=0.05, hspace=0.05, top=0.9, bottom=0.525)
# # gs4 = fig.add_gridspec(nrows=3, ncols=3, left=0.8, right=1, wspace=0.05, hspace=0.05, top=0.9, bottom=0.525)
# # gs5 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.25, wspace=0.05, hspace=0.05, top=0.475, bottom=0.1)
# # gs6 = fig.add_gridspec(nrows=3, ncols=3, left=0.25, right=0.45, wspace=0.05, hspace=0.05, top=0.475, bottom=0.1)
# # gs7 = fig.add_gridspec(nrows=3, ncols=3, left=0.6, right=0.8, wspace=0.05, hspace=0.05, top=0.475, bottom=0.1)
#
#
# count=0
# for i in range(0, 3):
#     for j in range(0, 3):
#
#         g1_ax = fig.add_subplot(gs1[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_6_random[count]) + ".png")
#         g1_ax.imshow(image)
#         g1_ax.get_xaxis().set_visible(False)
#         g1_ax.get_yaxis().set_visible(False)
#
#         g2_ax = fig.add_subplot(gs2[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_2_random[count]) + ".png")
#         g2_ax.imshow(image)
#         g2_ax.get_xaxis().set_visible(False)
#         g2_ax.get_yaxis().set_visible(False)
#
#         g3_ax = fig.add_subplot(gs3[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_3_random[count]) + ".png")
#         g3_ax.imshow(image)
#         g3_ax.get_xaxis().set_visible(False)
#         g3_ax.get_yaxis().set_visible(False)
#
#         g4_ax = fig.add_subplot(gs4[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_5_random[count]) + ".png")
#         g4_ax.imshow(image)
#         g4_ax.get_xaxis().set_visible(False)
#         g4_ax.get_yaxis().set_visible(False)
#
#         g5_ax = fig.add_subplot(gs5[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[count]) + ".png")
#         g5_ax.imshow(image)
#         g5_ax.get_xaxis().set_visible(False)
#         g5_ax.get_yaxis().set_visible(False)
#
#         g6_ax = fig.add_subplot(gs6[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_7_random[count]) + ".png")
#         g6_ax.imshow(image)
#         g6_ax.get_xaxis().set_visible(False)
#         g6_ax.get_yaxis().set_visible(False)
#
#         g7_ax = fig.add_subplot(gs7[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_4_random[count]) + ".png")
#         g7_ax.imshow(image)
#         g7_ax.get_xaxis().set_visible(False)
#         g7_ax.get_yaxis().set_visible(False)
#
#         # set group title for middle plot of each group
#         if i == 0 and j == 1:
#             g1_ax.set_title(("Group 1-1-1 (" + str(np.array(group_6).shape[0]) + ")"), fontsize=15, pad=20)
#             g2_ax.set_title(("Group 1-1-2 (" + str(np.array(group_2).shape[0]) + ")"), fontsize=15, pad=20)
#             g3_ax.set_title(("Group 2-1 (" + str(np.array(group_3).shape[0]) + ")"), fontsize=15, pad=20)
#             g4_ax.set_title(("Group 1-2-1 (" + str(np.array(group_5).shape[0]) + ")"), fontsize=15, pad=20)
#             g5_ax.set_title(("Group 1-2-2 (" + str(np.array(group_1).shape[0]) + ")"), fontsize=15, pad=20)
#             g6_ax.set_title(("Group 2-2-1 (" + str(np.array(group_7).shape[0]) + ")"), fontsize=15, pad=20)
#             g7_ax.set_title(("Group 2-2-2 (" + str(np.array(group_4).shape[0]) + ")"), fontsize=15, pad=20)
#
#         count += 1
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(encoding_dim) + "_feature_stripped")
# plt.show()




# # order = [1, 0]
# # order = [5, 6, 4, 0, 7, 3, 10, 2, 8, 9, 1]
# # order = [7, 8, 1, 4, 0, 6, 5, 2, 3]
# order = [4, 3, 12, 13, 11, 7, 9, 10, 5, 6, 1, 8, 2, 0]
#
#
# for i, cluster in enumerate(order):
#     print(i, cluster)
#
#     galaxy_ids = np.load("Clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(cluster) + ".npy")
#
#     fig, axs = plt.subplots(5, 5, figsize=(20, 20))
#
#     count = 0
#
#     for j in range(0, 5):
#         for k in range(0, 5):
#
#             image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxy_ids[count]) + ".png")
#
#             sersic = str(all_properties[all_properties["GalaxyID"] == galaxy_ids[count]]["n_r"].tolist()[0])
#             # axis_ratio = str(all_properties[all_properties["GalaxyID"] == galaxy_ids[count]]["q_r"].tolist()[0])
#             # stellar_mass = str(all_properties[all_properties["GalaxyID"] == galaxy_ids[count]]["MassType_Star"].tolist()[0])
#
#             axs[j, k].imshow(image)
#             axs[j, k].get_xaxis().set_visible(False)
#             axs[j, k].get_yaxis().set_visible(False)
#             axs[j, k].set_title(str(galaxy_ids[count]) + " " + str(sersic), fontsize=18)
#
#             count += 1
#
#     plt.savefig("Cluster Images/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters_" + str(i) + "_" + str(cluster))














# plt.rc("text", usetex=True)
#
# group_4 = [17068442, 16608247, 8915064, 15973150]
# group_3 = [11495444, 8133712, 9481539, 7680777]
# # group_12 = [9067175, 15528598, 9406027, 9782789]
# group_12 = [10323277, 15528598, 18129275, 9334220]
# group_13 = [12122358, 10690127, 14349778, 8733179]
# group_11 = [8667712, 615891, 17331332, 12663357]
# group_7 = [10701959, 14021166, 8109571, 14822490]
# group_9 = [16517497, 15242252, 17025268, 9753783]
# group_10 = [11079170, 9517737, 9865665, 14861046]
# group_5 = [9045697, 17731119, 10345837, 13250441]
# group_6 = [8854395, 10258829, 10425479, 9831288]
# group_1 = [17704712, 17154752, 16996938, 13190588]
# group_8 = [15595159, 12178298, 9147203, 8814843]  # tight spiral arms
# group_2 = [17691609, 11506546, 10223986, 14770895]  # very edge on (originally elliptical)
# group_0 = [17462825, 1732243, 13869651, 10174638] # irregular galaxies (barred spiral, merger)
#
#
# fig = plt.figure(constrained_layout=False, figsize=(15, 15))
#
# # create sub figures within main figure, specify their location
# gs1 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.975, bottom=0.775)
# gs2 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.975, bottom=0.775)
# gs3 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.975, bottom=0.775)
# gs4 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.975, bottom=0.775)
#
# gs5 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.725, bottom=0.525)
# gs6 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.725, bottom=0.525)
# gs7 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.725, bottom=0.525)
# gs8 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.725, bottom=0.525)
#
# gs9 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.475, bottom=0.275)
# gs10 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.475, bottom=0.275)
# gs11 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.475, bottom=0.275)
# gs12 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.475, bottom=0.275)
#
# gs13 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.025, right=0.225, top=0.225, bottom=0.025)
# gs14 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.275, right=0.475, top=0.225, bottom=0.025)
# gs15 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.525, right=0.725, top=0.225, bottom=0.025)
# gs16 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.05, hspace=0.05, left=0.775, right=0.975, top=0.225, bottom=0.025)
#
#
# count = 0
#
# for i in range(0, 2):
#     for j in range(0, 2):
#
#         g1_ax = fig.add_subplot(gs1[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_4[count]) + ".png")
#         g1_ax.imshow(image)
#         sersic = str(all_properties[all_properties["GalaxyID"] == group_4[count]]["n_r"].tolist()[0])
#         print(sersic)
#         g1_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_4[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g1_ax.set_xticks([])
#         g1_ax.get_yaxis().set_visible(False)
#
#         g2_ax = fig.add_subplot(gs2[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_3[count]) + ".png")
#         g2_ax.imshow(image)
#         g2_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_3[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g2_ax.set_xticks([])
#         g2_ax.get_yaxis().set_visible(False)
#
#         g3_ax = fig.add_subplot(gs3[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_12[count]) + ".png")
#         g3_ax.imshow(image)
#         g3_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_12[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g3_ax.set_xticks([])
#         g3_ax.get_yaxis().set_visible(False)
#
#         g4_ax = fig.add_subplot(gs4[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_13[count]) + ".png")
#         g4_ax.imshow(image)
#         g4_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_13[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g4_ax.set_xticks([])
#         g4_ax.get_yaxis().set_visible(False)
#
#         g5_ax = fig.add_subplot(gs5[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_11[count]) + ".png")
#         g5_ax.imshow(image)
#         g5_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_11[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g5_ax.set_xticks([])
#         g5_ax.get_yaxis().set_visible(False)
#
#         g6_ax = fig.add_subplot(gs6[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_7[count]) + ".png")
#         g6_ax.imshow(image)
#         g6_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_7[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g6_ax.set_xticks([])
#         g6_ax.get_yaxis().set_visible(False)
#
#         g7_ax = fig.add_subplot(gs7[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_9[count]) + ".png")
#         g7_ax.imshow(image)
#         g7_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_9[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g7_ax.set_xticks([])
#         g7_ax.get_yaxis().set_visible(False)
#
#         g8_ax = fig.add_subplot(gs8[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_10[count]) + ".png")
#         g8_ax.imshow(image)
#         g8_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_10[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g8_ax.set_xticks([])
#         g8_ax.get_yaxis().set_visible(False)
#
#         g9_ax = fig.add_subplot(gs9[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_5[count]) + ".png")
#         g9_ax.imshow(image)
#         g9_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_5[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g9_ax.set_xticks([])
#         g9_ax.get_yaxis().set_visible(False)
#
#         g10_ax = fig.add_subplot(gs10[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_6[count]) + ".png")
#         g10_ax.imshow(image)
#         g10_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_6[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g10_ax.set_xticks([])
#         g10_ax.get_yaxis().set_visible(False)
#
#         g11_ax = fig.add_subplot(gs11[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_1[count]) + ".png")
#         g11_ax.imshow(image)
#         g11_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_1[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g11_ax.set_xticks([])
#         g11_ax.get_yaxis().set_visible(False)
#
#         g12_ax = fig.add_subplot(gs12[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_8[count]) + ".png")
#         g12_ax.imshow(image)
#         g12_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_8[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g12_ax.set_xticks([])
#         g12_ax.get_yaxis().set_visible(False)
#
#         g13_ax = fig.add_subplot(gs13[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_2[count]) + ".png")
#         g13_ax.imshow(image)
#         g13_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_2[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g13_ax.set_xticks([])
#         g13_ax.get_yaxis().set_visible(False)
#
#         g14_ax = fig.add_subplot(gs14[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_0[count]) + ".png")
#         g14_ax.imshow(image)
#         g14_ax.set_xlabel("$n$ = " + str(np.round(all_properties[all_properties["GalaxyID"] == group_0[count]]["n_r"].tolist()[0], 2)), fontsize=15, labelpad=-18, color="white")
#         g14_ax.set_xticks([])
#         g14_ax.get_yaxis().set_visible(False)
#
#         count += 1
#
#
# ax1 = fig.add_subplot(gs1[:])
# ax1.axis("off")
# ax1.set_title("Group 4", fontsize=25)
#
# ax2 = fig.add_subplot(gs2[:])
# ax2.axis("off")
# ax2.set_title("Group 3", fontsize=25)
#
# ax3 = fig.add_subplot(gs3[:])
# ax3.axis("off")
# ax3.set_title("Group 12", fontsize=25)
#
# ax4 = fig.add_subplot(gs4[:])
# ax4.axis("off")
# ax4.set_title("Group 13", fontsize=25)
#
# ax5 = fig.add_subplot(gs5[:])
# ax5.axis("off")
# ax5.set_title("Group 11", fontsize=25)
#
# ax6 = fig.add_subplot(gs6[:])
# ax6.axis("off")
# ax6.set_title("Group 7", fontsize=25)
#
# ax7 = fig.add_subplot(gs7[:])
# ax7.axis("off")
# ax7.set_title("Group 9", fontsize=25)
#
# ax8 = fig.add_subplot(gs8[:])
# ax8.axis("off")
# ax8.set_title("Group 10", fontsize=25)
#
# ax9 = fig.add_subplot(gs9[:])
# ax9.axis("off")
# ax9.set_title("Group 5", fontsize=25)
#
# ax10 = fig.add_subplot(gs10[:])
# ax10.axis("off")
# ax10.set_title("Group 6", fontsize=25)
#
# ax11 = fig.add_subplot(gs11[:])
# ax11.axis("off")
# ax11.set_title("Group 1", fontsize=25)
#
# ax12 = fig.add_subplot(gs12[:])
# ax12.axis("off")
# ax12.set_title("Group 8", fontsize=25)
#
# ax13 = fig.add_subplot(gs13[:])
# ax13.axis("off")
# ax13.set_title("Group 2", fontsize=25)
#
# ax14 = fig.add_subplot(gs14[:])
# ax14.axis("off")
# ax14.set_title("Group 0", fontsize=25)
#
#
#
# plt.savefig("Cluster Images/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters/all_clusters", bbox_inches='tight')

