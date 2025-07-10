import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
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

# print(all_properties)
#
# selected_structure = ["DiscToTotal", "n_r", "q_r", "pa_r", "rhalf_ellip", "concentration", "asymmetry", "smoothness"]
# selected_physical = ["MassType_Star", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]
#
# selected_properties = selected_structure + selected_physical
#
# print(selected_properties)
#
# indices = [all_properties.columns.get_loc(property) for property in selected_properties]

reconstruction_indices = [780, 560, 743, 2227, 2785, 2929, 495, 437, 2581]

print(all_properties)
print(all_properties.loc[reconstruction_indices]["GalaxyID"].tolist())

[8827412, 8407169, 8756517, 13632283, 16618997, 17171464, 8274107, 8101596, 15583095]




