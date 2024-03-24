import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# import keras
import os
from matplotlib import image as mpimg
import seaborn as sns
import random


# A = [1, 2, 3]
# B = [4, 5, 6]
# C = [1, 2, 3]

A = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]

A = np.array(A)

B = A.T

C = np.array([list(B[1]), list(B[2])]).T

print(C)
