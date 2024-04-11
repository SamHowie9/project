import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.neighbors import NearestCentroid
from matplotlib import image as mpimg
import random
import cv2




# set the encoding dimension (number of extracted features)
encoding_dim = 38

# set the number of clusters
n_clusters = 14


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

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



# fig, axs = plt.subplots(4, 4, figsize=(20, 20))







# # create the figure for the plot
# fig = plt.figure(constrained_layout=False, figsize=(20, 10))
#
# # create the subfigures for the plot (each group)
# gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.45, wspace=0.05, hspace=0.05)
# gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.95, wspace=0.05, hspace=0.05)
#
# # # high sersic
# # galaxies_1 = [217859, 234331, 244671, 629180, 1008743, 1732243, 1774857, 2425267, 2446634]
# # # low sersic
# # galaxies_2 = [34157, 35658, 43262, 130678, 138061, 178838, 641392, 1049778, 2047699]
# #
# # # high stripped sersic
# # galaxies_3 = [50759, 65696, 68767, 246800, 966292, 1028772, 1406432, 1704972, 1738146]
# # # low stripped sersic
# # galaxies_4 = [1383229, 1427448, 2331971, 7182472, 13869651, 13985849, 14237115, 14402768, 15037053]
#
# galaxies_1 = [10733159, 8122788, 4518274, 9532695, 11533908, 8686633, 16677355, 3523446, 10733159]
# galaxies_2 = [11564243, 8536493, 8986033, 10806034, 16720414, 9187925, 9469843, 16420095, 9747706]
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
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxies_2[count]) + ".png")
#         g2_ax.imshow(image)
#         g2_ax.get_xaxis().set_visible(False)
#         g2_ax.get_yaxis().set_visible(False)
#
#         # set group title for middle plot of each group
#         if i == 0 and j == 1:
#             # g1_ax.set_title("High Sersic (Elliptical-Like)", fontsize=25, pad=20)
#             # g2_ax.set_title("Low Sersic (Spiral-Like)", fontsize=25, pad=20)
#             g1_ax.set_title("Less Featured (Elliptical-Like)", fontsize=25, pad=20)
#             g2_ax.set_title("More Featured (Spiral-Like)", fontsize=25, pad=20)
#
#         count += 1
#
#
#
# plt.savefig("Plots/" + str(n_clusters) + "_cluster_" + str(encoding_dim) + "_feature_originals", bbox_inches='tight')
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




# fig, axs = plt.subplots(3, 3, figsize=(20, 20))
#
# galaxies = [6066836, 8471322, 8860264, 9220513, 13857961, 16472250, 16623393, 16882281, 17462825]
#
# count = 0
#
# for i in range(0, 3):
#     for j in range(0, 3):
#
#         sersic = str(all_properties[all_properties["GalaxyID"] == galaxies[count]]["n_r"].tolist()[0])
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxies[count]) + ".png")
#
#         axs[i, j].imshow(image)
#         axs[i, j].get_xaxis().set_visible(False)
#         axs[i, j].get_yaxis().set_visible(False)
#         axs[i, j].set_title((str(galaxies[count]) + " " + str(sersic)), fontsize=18)
#
#         count += 1
#
# plt.savefig("Plots/double_fit_sersic_images")





# fig, axs = plt.subplots(4, 4, figsize=(20, 20))
#
# galaxies = [8606403, 8764213, 13176887, 13202231, 13989414, 14018346, 14042157, 14107961, 14427787, 15336992, 15656318, 16072005, 16169303, 17582605, 17733932]
#
# count = 0
#
# for i in range(0, 4):
#     for j in range(0, 4):
#
#         axs[i, j].get_xaxis().set_visible(False)
#         axs[i, j].get_yaxis().set_visible(False)
#
#         if count == 15:
#             break
#
#         sersic = str(all_properties[all_properties["GalaxyID"] == galaxies[count]]["n_r"].tolist()[0])
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxies[count]) + ".png")
#
#         axs[i, j].imshow(image)
#         axs[i, j].set_title((str(galaxies[count]) + " " + str(sersic)), fontsize=18)
#
#         count += 1
#
# plt.savefig("Plots/bad_fit_sersic_images")




# fig, axs = plt.subplots(4, 4, figsize=(20, 20))
#
# galaxies = [12202542, 14289611, 15358401, 15371869, 15511664, 15535362, 15978821, 16736005, 16921469, 17109600, 17413315, 17760224, 17805326, 18104018, 18135698]
#
# count = 0
#
# for i in range(0, 4):
#     for j in range(0, 4):
#
#         axs[i, j].get_xaxis().set_visible(False)
#         axs[i, j].get_yaxis().set_visible(False)
#
#         if count == 15:
#             break
#
#         sersic = str(all_properties[all_properties["GalaxyID"] == galaxies[count]]["n_r"].tolist()[0])
#
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(galaxies[count]) + ".png")
#
#         axs[i, j].imshow(image)
#         axs[i, j].set_title((str(galaxies[count]) + " " + str(sersic)), fontsize=18)
#
#         count += 1
#
# plt.savefig("Plots/unreasonable_sersic_images")





# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(16536187) + ".png")
# axs[0].imshow(image)
# axs[0].get_xaxis().set_visible(False)
# axs[0].get_yaxis().set_visible(False)
#
# image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(16643442) + ".png")
# axs[1].imshow(image)
# axs[1].get_xaxis().set_visible(False)
# axs[1].get_yaxis().set_visible(False)
#
# plt.savefig("Plots/flag_5")






group_4 = [17068442, 16608247, 8915064, 15973150]
group_3 = [11495444, 8133712, 9481539, 7680777]
group_12 = [9067175, 15528598, 9406027, 9782789]
group_13 = [12122358, 10690127, 14349778, 8733179]
group_11 = [8667712, 615891, 17331332, 12663357]
group_7 = [10701959, 14021166, 8109571, 14822490]
group_9 = [16517497, 15242252, 17025268, 9753783]
group_10 = [11079170, 9517737, 9865665, 14861046]
group_5 = [9045697, 17731119, 10345837, 13250441]
group_6 = [8854395, 10258829, 10425479, 9831288]
group_1 = [17704712, 17154752, 16996938, 13190588]
group_8 = [15595159, 12178298, 9147203, 8814843]  # tight spiral arms
group_2 = [17691609, 11506546, 10223986, 14770895]  # very edge on (originally elliptical)
group_0 = [17462825, 1732243, 13869651, 10174638] # irregular galaxies (barred spiral, merger)


fig = plt.figure(constrained_layout=False, figsize=(20, 20))

# create sub figures within main figure, specify their location
gs1 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.025, right=0.225, top=0.975, bottom=0.775)
gs2 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.275, right=0.475, top=0.975, bottom=0.775)
gs3 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.525, right=0.725, top=0.975, bottom=0.775)
gs4 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.775, right=0.975, top=0.975, bottom=0.775)

gs5 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.025, right=0.225, top=0.725, bottom=0.525)
gs6 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.275, right=0.475, top=0.725, bottom=0.525)
gs7 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.525, right=0.725, top=0.725, bottom=0.525)
gs8 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.775, right=0.975, top=0.725, bottom=0.525)

gs9 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.025, right=0.225, top=0.475, bottom=0.275)
gs10 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.275, right=0.475, top=0.475, bottom=0.275)
gs11 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.525, right=0.725, top=0.475, bottom=0.275)
gs12 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.775, right=0.975, top=0.475, bottom=0.275)

gs13 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.025, right=0.225, top=0.225, bottom=0.025)
gs14 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.275, right=0.475, top=0.225, bottom=0.025)
gs15 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.525, right=0.725, top=0.225, bottom=0.025)
gs16 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1, hspace=0.1, left=0.775, right=0.975, top=0.225, bottom=0.025)


count = 0

for i in range(0, 2):
    for j in range(0, 2):

        g1_ax = fig.add_subplot(gs1[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_4[count]) + ".png")
        g1_ax.imshow(image)
        g1_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_4[count]]["n_r"].tolist()[0]), fontsize=20)
        # g1_ax.get_xaxis().set_visible(False)
        g1_ax.get_yaxis().set_visible(False)

        g2_ax = fig.add_subplot(gs2[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_3[count]) + ".png")
        g2_ax.imshow(image)
        g2_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_3[count]]["n_r"].tolist()[0]), fontsize=20)
        # g2_ax.get_xaxis().set_visible(False)
        g2_ax.get_yaxis().set_visible(False)

        g3_ax = fig.add_subplot(gs3[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_12[count]) + ".png")
        g3_ax.imshow(image)
        g3_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_12[count]]["n_r"].tolist()[0]), fontsize=20)
        # g3_ax.get_xaxis().set_visible(False)
        g3_ax.get_yaxis().set_visible(False)

        g4_ax = fig.add_subplot(gs4[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_13[count]) + ".png")
        g4_ax.imshow(image)
        g4_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_13[count]]["n_r"].tolist()[0]), fontsize=20)
        # g4_ax.get_xaxis().set_visible(False)
        g4_ax.get_yaxis().set_visible(False)

        g5_ax = fig.add_subplot(gs5[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_11[count]) + ".png")
        g5_ax.imshow(image)
        g5_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_11[count]]["n_r"].tolist()[0]), fontsize=20)
        # g5_ax.get_xaxis().set_visible(False)
        g5_ax.get_yaxis().set_visible(False)

        g6_ax = fig.add_subplot(gs6[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_7[count]) + ".png")
        g6_ax.imshow(image)
        g6_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_7[count]]["n_r"].tolist()[0]), fontsize=20)
        # g6_ax.get_xaxis().set_visible(False)
        g6_ax.get_yaxis().set_visible(False)

        g7_ax = fig.add_subplot(gs7[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_9[count]) + ".png")
        g7_ax.imshow(image)
        g7_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_9[count]]["n_r"].tolist()[0]), fontsize=20)
        # g7_ax.get_xaxis().set_visible(False)
        g7_ax.get_yaxis().set_visible(False)

        g8_ax = fig.add_subplot(gs8[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_10[count]) + ".png")
        g8_ax.imshow(image)
        g8_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_10[count]]["n_r"].tolist()[0]), fontsize=20)
        # g8_ax.get_xaxis().set_visible(False)
        g8_ax.get_yaxis().set_visible(False)

        g9_ax = fig.add_subplot(gs9[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_5[count]) + ".png")
        g9_ax.imshow(image)
        g9_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_5[count]]["n_r"].tolist()[0]), fontsize=20)
        # g9_ax.get_xaxis().set_visible(False)
        g9_ax.get_yaxis().set_visible(False)

        g10_ax = fig.add_subplot(gs10[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_6[count]) + ".png")
        g10_ax.imshow(image)
        g10_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_6[count]]["n_r"].tolist()[0]), fontsize=20)
        # g10_ax.get_xaxis().set_visible(False)
        g10_ax.get_yaxis().set_visible(False)

        g11_ax = fig.add_subplot(gs11[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_1[count]) + ".png")
        g11_ax.imshow(image)
        g11_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_1[count]]["n_r"].tolist()[0]), fontsize=20)
        # g11_ax.get_xaxis().set_visible(False)
        g11_ax.get_yaxis().set_visible(False)

        g12_ax = fig.add_subplot(gs12[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_8[count]) + ".png")
        g12_ax.imshow(image)
        g12_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_8[count]]["n_r"].tolist()[0]), fontsize=20)
        # g12_ax.get_xaxis().set_visible(False)
        g12_ax.get_yaxis().set_visible(False)

        g13_ax = fig.add_subplot(gs13[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_2[count]) + ".png")
        g13_ax.imshow(image)
        g13_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_2[count]]["n_r"].tolist()[0]), fontsize=20)
        # g13_ax.get_xaxis().set_visible(False)
        g13_ax.get_yaxis().set_visible(False)

        g14_ax = fig.add_subplot(gs14[i, j])
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galrand_" + str(group_0[count]) + ".png")
        g14_ax.imshow(image)
        g14_ax.set_xlabel("n = " + str(all_properties[all_properties["GalaxyID"] == group_0[count]]["n_r"].tolist()[0]), fontsize=20)
        # g14_ax.get_xaxis().set_visible(False)
        g14_ax.get_yaxis().set_visible(False)

        count += 1

plt.savefig("Cluster Images/" + str(encoding_dim) + "_features_" + str(n_clusters) + "_clusters/all_clusters", bbox_inches='tight')

