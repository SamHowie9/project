import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import ops
from keras.layers import Layer, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import random


# for run_number, (beta, filename) in enumerate([[0.001, "001"], [0.0001, "0001"], [0.00001, "00001"], [0.000001, "000001"], [0.0000001, "0000001"]]):

betas = [0.001, 0.001, 0.00001, 0.000001, 0.0000001]

reconstruction_loss = np.load("Variational Eagle/Loss/test")

fig, axs = plt.subplots(3, 1, figsize=(12, 15))



