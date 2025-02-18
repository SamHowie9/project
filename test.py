import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random
# import cv2
# import keras
# from keras import ops
# from tensorflow.keras import backend as K
# import tensorflow as tf



# A = [[1, 2], [3, 4], [5, 6], [7, 8]]
# B = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# B = [[3, 3], [5, 5], [9, 9], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4], [4, 4]]

A = np.array([10, 12, 414, 62, 62, 8, 9])
B = np.array([1, 2, 3, 4, 5, 6, 7])



i = 0

for j in range(1, 100):
    print(".")
    i += 1
    if i >= 5:
        break