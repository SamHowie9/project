import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf
# os.environ['NUMBA_DISABLE_CACHE'] = '1'


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

print(all_properties)

selected_structure = ["DiscToTotal", "pa_r", "rhalf_ellip", "n_r", "q_r", "concentration", "asymmetry", "smoothness"]
selected_physical = ["MassType_Star", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]

selected_properties = selected_structure + selected_physical

print(selected_properties)

indices = [all_properties.columns.get_loc(property) for property in selected_properties]

print(indices)

# print(all_properties[selected_properties])

row = all_properties.iloc[0].tolist()

print(all_properties[selected_properties])

print([row[index] for index in indices])

print(row)