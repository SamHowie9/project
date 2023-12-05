import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import keras
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns
from sklearn.cluster import KMeans


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




# compile the autoencoder model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# load the weights
autoencoder.load_weights("Weights/8_feature_weights.h5")





# load the extracted features
extracted_features = np.load("Features/8_features.npy")



kmeans = KMeans(n_clusters=8, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)

centers = kmeans.cluster_centers_
# print(centers)

for center in centers:
    print(center)

# print(centers[0])

reconstructions = decoder.predict(centers)

print(reconstructions[0])

# create figure to hold subplots
fig, axs = plt.subplots(2, 4, figsize=(20,4))

axs[0, 0].imshow(np.exp(5 * reconstructions[0]) - 1)
axs[0, 1].imshow(np.exp(5 * reconstructions[1]) - 1)
axs[0, 2].imshow(np.exp(5 * reconstructions[2]) - 1)
axs[0, 3].imshow(np.exp(5 * reconstructions[3]) - 1)
axs[1, 0].imshow(np.exp(5 * reconstructions[4]) - 1)
axs[1, 1].imshow(np.exp(5 * reconstructions[5]) - 1)
axs[1, 2].imshow(np.exp(5 * reconstructions[6]) - 1)
axs[1, 3].imshow(np.exp(5 * reconstructions[7]) - 1)

plt.show()


# df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7"])
# df["Category"] = clusters


# print(df)

# kws = dict(s=5, linewidth=0)
#
# sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")
#
# plt.savefig("Plots/7_feature_clustering")
# plt.show()

