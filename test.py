import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from matplotlib import image as mpimg


# face = [0, 0, 0, 0, 0, 0]
# edge = [0, 0, 0, 0, 0, 0]
# rand = [0, 0, 0, 0, 0, 0]



df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")


# print(df["GalaxyID"].tolist())
#
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

all_images = []

for i, galaxy in enumerate(df["GalaxyID"].tolist()):
    print(galaxy)
    try:
        image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/galface_" + str(galaxy) + ".png")
        all_images.append(image)
    except:
        print(str(galaxy), "...")
        print(".................................................................")
        df.drop(axis=0, index=i)
        i -= 1



# [1, 1, 0, 3624, 1, 1]
# [1, 1, 0, 3624, 1, 1]
# [1, 1, 0, 3624, 1, 1]