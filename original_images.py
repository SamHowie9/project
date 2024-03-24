import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random
import cv2




# set the encoding dimension (number of extracted features)
encoding_dim = 28

# set the number of clusters
n_clusters = 10




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
#
#
#
#
# def center_crop(img, dim):
#     width, height = img.shape[1], img.shape[0]
#     # process crop width and height for max available dimension
#     crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
#     crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
#
#     mid_x, mid_y = int(width / 2), int(height / 2)
#     cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
#     crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
#     return crop_img
#
#
#
# def half_max_range(image):
#
#     mean_intensity = image.mean()
#
#     intensity_x = image.mean(axis=2).mean(axis=0)
#     intensity_y = image.mean(axis=2).mean(axis=1)
#
#     half_max_intensity_x = np.max(intensity_x/mean_intensity) / 2
#     half_max_intensity_y = np.max(intensity_y/mean_intensity) / 2
#
#
#     size = len(intensity_x)
#
#     start_x = 0
#     start_y = 0
#     end_x = 255
#     end_y = 255
#
#     found_start_x = False
#     found_start_y = False
#     found_end_x = False
#     found_end_y = False
#
#     # loop through half of the image
#     for j in range(0, int(size / 2)):
#
#
#         # if we haven't previously found the cutoff point and are still below the cutoff, increment the pointer
#         if (found_start_x is False) and ((intensity_x[j] / mean_intensity) < half_max_intensity_x):
#             start_x += 1
#         else:
#             found_start_x = True
#
#         if (found_end_x is False) and ((intensity_x[-j] / mean_intensity) < half_max_intensity_x):
#             end_x -= 1
#         else:
#             found_end_x = True
#
#         if (found_start_y is False) and ((intensity_y[j] / mean_intensity) < half_max_intensity_y):
#             start_y += 1
#         else:
#             found_start_y = True
#
#         if (found_end_y is False) and ((intensity_y[-j] / mean_intensity) < half_max_intensity_y):
#             end_y -= 1
#         else:
#             found_end_y = True
#
#     return start_x, end_x, start_y, end_y
#
#
#
# def resize_image(image, cutoff=60):
#
#     # get the fill width half maximum (for x and y direction)
#     start_x, end_x, start_y, end_y = half_max_range(image)
#
#     # calculate the full width half maximum
#     range_x = end_x - start_x
#     range_y = end_y - start_y
#
#     # check if the majority of out image is within the cutoff range, if so, center crop, otherwise, scale image down
#     if (range_x <= cutoff) and (range_y <= cutoff):
#         image = center_crop(image, (128, 128))
#     else:
#         image = cv2.resize(image, (128, 128))
#
#     # return the resized image
#     return image
#
#
#
#
#
#
# # load the two excel files into dataframes
# df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")
# df2 = pd.read_csv("stab3510_supplemental_file/table2.csv", comment="#")
#
# # account for hte validation data and remove final 200 elements
# df1.drop(df1.tail(200).index, inplace=True)
# df2.drop(df2.tail(200).index, inplace=True)
#
# # extract relevant properties
# galaxy_id = df1["GalaxyID"]
# ab_magnitude = df1["galfit_mag"]
# mass = df2["galfit_lmstar"]
# semi_major = (df1["galfit_re"] + df2["galfit_re"]) / 2
# sersic = (df1["galfit_n"] + df2["galfit_n"]) / 2
# axis_ratio = (df1["galfit_q"] + df2["galfit_q"]) / 2
# position_angle = (df1["galfit_PA"] + df2["galfit_PA"]) / 2
#
# # create a new dataframe to contain all the relevant information about each galaxy
# df = pd.DataFrame(columns=["GalaxyID", "galfit_mag", "galfit_lmstar", "galfit_re", "galfit_n", "galfit_q", "galfit_PA", "Cluster"])
# df["GalaxyID"] = galaxy_id
# df["galfit_mag"] = ab_magnitude
# df["galfit_lmstar"] = mass
# df["galfit_re"] = semi_major
# df["galfit_n"] = sersic
# df["galfit_q"] = axis_ratio
# df["galfit_PA"] = position_angle
# df["Cluster"] = clusters
#
#
#
#
#
# separate the dataframe into groups based on their cluster
# group_1 = df.loc[df["Cluster"] == 0]
# group_2 = df.loc[df["Cluster"] == 1]
#
# # get a list of 9 random indices for each group
# group_1_random_index = random.sample(range(0, len(group_1)), 25)
# group_2_random_index = random.sample(range(0, len(group_2)), 25)
#
# # get the galaxy id of each of the random indices for each group
# group_1_random = group_1["GalaxyID"].iloc[group_1_random_index].tolist()
# group_2_random = group_2["GalaxyID"].iloc[group_2_random_index].tolist()
#
# print(group_1_random)
# print(group_2_random)



fig, axs = plt.subplots(4, 4, figsize=(20, 20))






# # create the figure for the plot
# fig = plt.figure(constrained_layout=False, figsize=(20, 10))
#
# # create the subfigures for the plot (each group)
# gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05)
# gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05)
#
# # high sersic
# galaxies_1 = [217859, 234331, 244671, 629180, 1008743, 1732243, 1774857, 2425267, 2446634]
# # low sersic
# galaxies_2 = [34157, 35658, 43262, 130678, 138061, 178838, 641392, 1049778, 2047699]
#
# # high stripped sersic
# galaxies_3 = [50759, 65696, 68767, 246800, 966292, 1028772, 1406432, 1704972, 1738146]
# # low stripped sersic
# galaxies_4 = [1383229, 1427448, 2331971, 7182472, 13869651, 13985849, 14237115, 14402768, 15037053]
#
#
# count = 0
#
# for i in range(0, 3):
#     for j in range(0, 3):
#
#         g1_ax = fig.add_subplot(gs1[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxies_1[count]) + ".png")
#         g1_ax.imshow(image)
#         g1_ax.get_xaxis().set_visible(False)
#         g1_ax.get_yaxis().set_visible(False)
#
#         g2_ax = fig.add_subplot(gs2[i, j])
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxies_2[count]) + ".png")
#         g2_ax.imshow(image)
#         g2_ax.get_xaxis().set_visible(False)
#         g2_ax.get_yaxis().set_visible(False)
#
#         # set group title for middle plot of each group
#         if i == 0 and j == 1:
#             # g1_ax.set_title("High Sersic (Elliptical-Like)", fontsize=25, pad=20)
#             # g2_ax.set_title("Low Sersic (Spiral-Like)", fontsize=25, pad=20)
#             g1_ax.set_title("High Sersic Index (Elliptical-Like)", fontsize=25, pad=20)
#             g2_ax.set_title("Low Sersic Index (Spiral-Like)", fontsize=25, pad=20)
#
#         count += 1
#
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(encoding_dim) + "_feature_sersic")
# plt.show()






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
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(encoding_dim) + "_feature_stripped")
# plt.show()


order = [4, 6, 0, 2, 1, 3, 9, 8, 7, 5]

for i, cluster in enumerate(order):
    print(i, cluster)

    galaxy_ids = np.load("Clusters/" + str(n_clusters) + "_cluster_" + str(cluster))

    fig, axs = plt.subplots(5, 5, figsize=(20, 20))

    count = 0

    for j in range(0, 5):
        for k in range(0, 5):

            image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxy_ids[count]) + ".png")

            axs[j, k].imshow(image)
            axs[j, k].get_xaxis().set_visible(False)
            axs[j, k].get_yaxis().set_visible(False)
            axs[k, k].set_title(galaxy_ids[count])

            count += 1

    plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(i) + "_" + str(cluster))
