from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import tensorflow as tf
import keras
from keras import ops
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2

os.environ["KERAS_BACKEND"] = "tensorflow"


# create sampling layer
class Sampling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mea)[1]
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon



# number of extracted features
encoding_dim = 38

# Define keras tensor for the encoder
input_image = keras.Input(shape=(128, 128, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 64, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=64, activation="relu")(x)                                                           # (64)
z_mean = Dense(encoding_dim, name="z_mean")(x)
z_log_var = Dense(encoding_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

# build the encoder
encoder = keras.Model(input_image, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

