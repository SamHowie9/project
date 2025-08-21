import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random

from matplotlib.pyplot import figure
from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)






# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")
# print(all_properties[all_properties["GalaxyID"] == 18365401].index.values[0])
# print(all_properties[all_properties["GalaxyID"] == 8827412].index.values[0])
# print(all_properties[all_properties["GalaxyID"] == 8937440].index.values[0])
# print(all_properties[all_properties["GalaxyID"] == 8407169].index.values[0])
# print(all_properties[all_properties["GalaxyID"] == 8756517].index.values[0])
#
# print()

all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_spirals.csv")
print(all_properties[all_properties["GalaxyID"] == 18365401].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 8827412].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 8937440].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 8407169].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 8756517].index.values[0])

print()

all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_transitional.csv")
print(all_properties[all_properties["GalaxyID"] == 13715538].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 13632283].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 18481115].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 16618997].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 17171464].index.values[0])

print()

all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_ellipticals.csv")
print(all_properties[all_properties["GalaxyID"] == 8274107].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 10042584].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 8101596].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 15583095].index.values[0])
print(all_properties[all_properties["GalaxyID"] == 17523255].index.values[0])
