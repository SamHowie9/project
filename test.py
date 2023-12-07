import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from matplotlib import image as mpimg
import seaborn as sns
import random


image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(group_1_random[i]) + ".png")
plt.imshow(image)
plt.savefig("Plots/test")



# # load the extracted features
# extracted_features = np.load("Features/16_features.npy")
#
#
# # add the features to a dataframe
# df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"])
#
#
#
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
#
# plt.savefig("Plots/16_feature_histogram")
# plt.show()





# A = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
# B = np.rot90(A)
# C = np.flipud(B)
# D = np.flipud(np.rot90(A))
# print(A)
# print()
# print(B)
# print()
# print(C)
# print()
# print(D)
# print()
#
# print(A.shape)
# print(B.shape)
# print(C.shape)
# print(D.shape)
#
#
# face = [0, 0, 0, 0, 0, 0]
# edge = [0, 0, 0, 0, 0, 0]
# rand = [0, 0, 0, 0, 0, 0]
#
#
#
# df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")


# A = ["a", "b", "c", "d", "e", "f"]
#
# B = ["b", "c", "d", "f"]
#
# dataset_df = pd.DataFrame(A, columns=["Letter"])
#
# print(dataset_df)
# print()
#
# for i, letter in enumerate(dataset_df["Letter"].tolist()):
#
#     print(i, letter)
#
#     if letter in B:
#         print(".")
#     else:
#         print("...")
#
#     # if letter not in B:
#     #     print(".....")
#     #     dataset_df.drop(axis=0, index=i, inplace=True)

# print(dataset_df)

# print(df["GalaxyID"].tolist())

# # loop through each galaxy set (with an index)
# for i, galaxy_set in enumerate(os.listdir("/cosma7/data/Eagle/web-storage/")):
#
#     # print(i, galaxy_set)
#
#     # loop through each file in that set
#     for file in os.listdir("/cosma7/data/Eagle/web-storage/" + galaxy_set + "/"):
#
#         # loop through each galaxy in the excel file
#         for galaxy in df["GalaxyID"].tolist():
#
#             if file == ("galface_" + str(galaxy) + ".png"):
#                 face[i] += 1
#
#             if file == ("galedge_" + str(galaxy) + ".png"):
#                 edge[i] += 1
#
#             if file == ("galrand_" + str(galaxy) + ".png"):
#                 rand[i] += 1
#
# print(face)
# print(edge)
# print(rand)

# pd.set_option('display.max_rows', None)

# print(df)

# print(df.shape)
#
# all_images = []
#
# for i, galaxy in enumerate(df["GalaxyID"].tolist()):
#     # print(galaxy)
#     # try:
#     #     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxy) + ".png")
#     #     all_images.append(image)
#     # except:
#     #     print(str(galaxy), "...")
#     #     print(".................................................................")
#     #     df.drop(axis=0, index=i)
#     #     i -= 1
#
#     filename = "galface_" + str(galaxy) + ".png"
#
#     if filename in os.listdir("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/"):
#         image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
#         all_images.append(image)
#     else:
#         df.drop(axis=0, index=i)
#         print(galaxy)
#
# print(df.shape)
# print(np.array(all_images).shape)

# print(df.shape)

# [1, 1, 0, 3624, 1, 1]
# [1, 1, 0, 3624, 1, 1]
# [1, 1, 0, 3624, 1, 1]