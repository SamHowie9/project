from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
from keras import backend as K
import numpy as np
# import IPython
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2


# set the encoding dimension (number of extracted features)
encoding_dim = 44


# Define keras tensor for the encoder
input_image = keras.Input(shape=(128, 128, 3))                                                      # (256, 256, 3)

# layers for the encoder
x = Conv2D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(input_image)    # (128, 128, 64)
x = Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (64, 128, 32)
x = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)              # (32, 32, 16)
x = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (16, 16, 8)
x = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)               # (8, 8, 4)
x = Flatten()(x)                                                                                    # (256)
x = Dense(units=64)(x)                                                                              # (32)
encoded = Dense(units=encoding_dim, name="encoded")(x)                                              # (2)


# layers for the decoder
x = Dense(units=64)(encoded)                                                                        # (32)
x = Dense(units=256)(x)                                                                             # (256)
x = Reshape((8, 8, 4))(x)                                                                           # (8, 8, 4)
x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (16, 16, 4)
x = Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(x)      # (32, 32, 8)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (64, 64, 16)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (128, 128, 32)
# x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)     # (256, 256, 64)
decoded = Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same", name="decoded")(x)        # (256, 256, 3)


# crate autoencoder
autoencoder = keras.Model(input_image, decoded)
autoencoder.summary()



# create the encoder using the autoencoder layers
encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])
encoder.summary()

print()

# create the decoder using the autoencoder layers
decoder = keras.Sequential()
for i in range(9, 17):
    decoder.add(autoencoder.layers[i])

print()

# build the decoder
decoder.build(input_shape=(None, encoding_dim))
decoder.summary()



# root means squared loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss=root_mean_squared_error)

# load the weights
autoencoder.load_weights("Weights/" + str(encoding_dim) + "_feature_weights.h5")



# load the extracted features
extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")
extracted_features_switch = np.flipud(np.rot90(extracted_features))



# keras.utils.plot_model(model=autoencoder, to_file="Plots/autoencoder_layers.png")




median_features = []
for i in range(encoding_dim):
    median_features.append(np.median(extracted_features_switch[i]))
# median_features = np.array(median_features)


# number of latent images per feature
latent_num = 15

# prepare the latent images for every feature
latent_features = []

# loop through every feature
for i in range(encoding_dim):

    latent_images = []

    # sample equally spaced values of that feature
    feature_values = np.linspace(min(extracted_features_switch[i]), max(extracted_features_switch[i]), latent_num)

    # loop through each image for that feature
    for j in range(latent_num):

        # create a list of the features which make up each image for that feature
        latent_image_features = median_features[:]
        latent_image_features[i] = feature_values[j]
        latent_images.append(latent_image_features)

    latent_features.append(latent_images)

latent_features = np.array(latent_features)






a = [median_features, median_features]

image = decoder.predict(np.array(a))

# fig, axs = plt.subplots(encoding_dim, latent_num, figsize=(10, 20))

width = latent_num + 2
height = encoding_dim + ((encoding_dim - 1) * 0.1) + 2

fig = plt.figure(constrained_layout=False, figsize=(width*2, height*2))

gs = fig.add_gridspec(nrows=encoding_dim, ncols=latent_num, hspace=0.1, wspace=0, left=(1/width), right=(1 - 1/width), bottom=(1/height), top=(1 - 1/height))


# fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

for i in range(encoding_dim):

    latent_images = decoder.predict(latent_features[i])

    for j in range(latent_num):

        ax = fig.add_subplot(gs[i, j])

        ax.imshow(latent_images[j])

        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_xticks([])


        # ax.get_yaxis().set_visible(False)
        # ax.get_xaxis().set_visible(False)


        if j == 0:
            ax.set_ylabel(i, fontsize=50, rotation=0, labelpad=40)

        # axs[i][j].imshow(latent_images[j], aspect="equal")
        # axs[i][j].get_xaxis().set_visible(False)
        # axs[i][j].get_yaxis().set_visible(False)




plt.savefig("Latent Plots/latent_" + str(encoding_dim) + "_features")

# print()
# print(latent_features[0][0].tolist())
# print(latent_features[0][4].tolist())


# plt.show()
