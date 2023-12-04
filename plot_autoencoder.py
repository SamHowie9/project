import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras



# set the encoding dimension (number of extracted features)
encoding_dim = 8

# Define keras tensor for the encoder
input_image = keras.Input(shape=(256, 256, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=32)(x)                                                                              # (32)
encoded = Dense(units=encoding_dim, name="encoded")(x)                                                         # (2)


# layers for the decoder
x = Dense(units=32)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)



# crate autoencoder
autoencoder = keras.Model(input_image, decoded)

# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# load the weights
autoencoder.load_weights("Weights/8_feature_weights.h5")

# save the weights
autoencoder.save_weights(filepath="Weights/8_feature_weights.h5", overwrite=True)



# extract encoder layer and decoder layer from autoencoder
encoder_layer = autoencoder.get_layer("encoded")
decoder_layer = autoencoder.get_layer("decoded")

# get the shape of the decoder input
decoder_input = keras.Input(shape=encoding_dim)

# build the encoder
encoder = keras.Model(autoencoder.input, encoder_layer.output)

# build the decoder
decoder = keras.Model(decoder_input, autoencoder.layers[-1](decoder_input))



# extract the features
extracted_features = encoder.predict(train_images)

# save the features as a numpy array
np.save("Features/8_features.npy", extracted_features)