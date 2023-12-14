from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd


# stores an empty list to contain all the image data to train the model
all_images = []

# load the supplemental file into a dataframe
df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

# loop through each galaxy in the supplemental file
for i, galaxy in enumerate(df["GalaxyID"].tolist()):

    # get the filename for that galaxy
    filename = "galface_" + str(galaxy) + ".png"

    # open the image and append it to the main list
    image = mpimg.imread("/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo/" + filename)
    all_images.append(image)