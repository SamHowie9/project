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


for i, (beta, filename) in enumerate([[0.001, "001"], [0.0001, "0001"], [0.00001, "00001"], [0.000001, "000001"], [0.0000001, "0000001"]]):
    # print(beta)
    print(i, beta, filename)
