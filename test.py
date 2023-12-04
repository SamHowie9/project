import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import os


count = [0, 0, 0, 0, 0, 0]


df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

print(df["GalaxyID"].tolist())

# loop through each galaxy set (with an index)
for i, galaxy_set in enumerate(os.listdir("/cosma7/data/Eagle/web-storage/")):

    # print(i, galaxy_set)

    # loop through each file in that set
    for file in os.listdir("/cosma7/data/Eagle/web-storage/" + galaxy_set + "/"):

        # loop through each galaxy in the excel file
        for galaxy in df["GalaxyID"].tolist():

            if file == ("galface_" + str(galaxy) + ".png"):

                count[i] += 1

print(count)
