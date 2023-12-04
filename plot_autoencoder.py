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

encoder = keras.Sequential()
for i in range(0, 9):
    encoder.add(autoencoder.layers[i])

decoder = keras.Sequential()
for i in range(9, 18):
    decoder.add(autoencoder.layers[i])

decoder.build(input_shape=(None, encoding_dim))

encoder.summary()
decoder.summary()


# # extract layers to build encoder
# encoded = autoencoder.layers[1]
# encoded = autoencoder.layers[2]
# encoded = autoencoder.layers[3]
# encoded = autoencoder.layers[4]
# encoded = autoencoder.layers[5]
# encoded = autoencoder.layers[6]
# encoded = autoencoder.layers[7]
# encoded = autoencoder.layers[8]
#
# encoder = keras.Model(input_image, encoded.output)
# encoder.summary()
#
# # input for decoder
# encoded_input = keras.Input(shape=(encoding_dim,))
#
# # extract layers for decoder
# decoded = autoencoder.layers[-1]
# decoded = autoencoder.layers[-2]
# decoded = autoencoder.layers[-3]
# decoded = autoencoder.layers[-4]
# decoded = autoencoder.layers[-5]
# decoded = autoencoder.layers[-6]
# decoded = autoencoder.layers[-7]
# decoded = autoencoder.layers[-8]
#
# decoder = keras.Model(encoded_input, autoencoder.layers[-9](encoded_input))
# decoder.add(autoencoder.layers[-8])
# decoder.summary()


# # extract encoder layer and decoder layer from autoencoder
# encoder_layer = autoencoder.get_layer("encoded")
# decoder_layer = autoencoder.get_layer("decoded")
#
#
#
# # build the encoder
# encoder = keras.Model(autoencoder.input, encoder_layer.output)
#
# # build the decoder
# encoded_input = keras.Input(shape=(8,))
# # Retrieve the last layer of the autoencoder model
# # decoder_layer = autoencoder.layers[-2]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))



# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# load the weights
autoencoder.load_weights("Weights/8_feature_weights.h5")




# load the extracted features
extracted_features = np.load("Features/8_features.npy")


# add the features to a dataframe
df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])





# create the pairplot with custom marker size
kws = dict(s=10)
g = sns.pairplot(df, corner=True, plot_kws=kws)


# function to add the correlation coefficient to the plots
def corrfunc(x, y, ax=None, color=None):
    # find the correlation coefficient and round to 3 dp
    correlation = np.corrcoef(x, y)[0][1]
    correlation = np.round(correlation, decimals=3)

    # annotate the plot with the correlation coefficient
    ax = ax or plt.gca()
    ax.annotate(("ρ = " + str(correlation)), xy=(0.1, 1), xycoords=ax.transAxes)


# add the correlation coefficient to each of the scatter plots
g.map_lower(corrfunc)

# add some vertical space between the plots (given we are adding the correlation coefficients
plt.subplots_adjust(hspace=0.2)



# save and display the plot
plt.savefig("Plots/8_feature_histogram_comparison")
plt.show()