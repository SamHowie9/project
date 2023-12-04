import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras





df = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")

print(df["GalaxyID"].tolist())

for galaxy_set, i in enumerate(os.listdir("/cosma7/data/Eagle/web-storage/")):
    print(i, galaxy_set)

