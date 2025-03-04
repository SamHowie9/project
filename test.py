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


def transformString(input):
    output = ""
    input = input.lower()

    for char in input:
        if input.count(char) == 1:
            output += "x"
        else:
            output += "y"

    return output

print(transformString("Hello"))