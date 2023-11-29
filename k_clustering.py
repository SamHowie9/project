import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


# # stores an empty list to contain all the image data to train the model
# all_images = []
#
# # loop through the directory containing all the image files
# for file in os.listdir("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/"):
#
#     # open the fits file and get the image data (this is a numpy array of each pixel value)
#     image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0025N0376_Subhalo/" + file)
#
#     # append the image data to the main list containing all data of all the images
#     all_images.append(image)



# open the file to load the extracted features
f = open("Features/6_features.txt", "r")
extracted_features = f.read()

print(extracted_features)



