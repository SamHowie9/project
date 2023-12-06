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
encoding_dim = 9

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
autoencoder.load_weights("Weights/9_feature_weights_new.h5")


# load the extracted features
extracted_features = np.load("Features/9_features_new.npy")




kmeans = KMeans(n_clusters=9, random_state=0, n_init='auto')
clusters = kmeans.fit_predict(extracted_features)

print(clusters)
print(clusters.shape)




# centers = kmeans.cluster_centers_
# # print(centers)
#
# for center in centers:
#     print(center)
#
# # print(centers[0])
#
# reconstructions = decoder.predict(centers)
#
# print(reconstructions[0])
#
# # create figure to hold subplots
# fig, axs = plt.subplots(2, 5, figsize=(20,10))
#
# axs[0, 0].imshow(reconstructions[0])
# axs[0, 0].get_xaxis().set_visible(False)
# axs[0, 0].get_yaxis().set_visible(False)
#
# axs[0, 1].imshow(reconstructions[1])
# axs[0, 1].get_xaxis().set_visible(False)
# axs[0, 1].get_yaxis().set_visible(False)
#
# axs[0, 2].imshow(reconstructions[2])
# axs[0, 2].get_xaxis().set_visible(False)
# axs[0, 2].get_yaxis().set_visible(False)
#
# axs[0, 3].imshow(reconstructions[3])
# axs[0, 3].get_xaxis().set_visible(False)
# axs[0, 3].get_yaxis().set_visible(False)
#
# axs[0, 4].imshow(reconstructions[4])
# axs[0, 4].get_xaxis().set_visible(False)
# axs[0, 4].get_yaxis().set_visible(False)
#
# axs[1, 0].imshow(reconstructions[5])
# axs[1, 0].get_xaxis().set_visible(False)
# axs[1, 0].get_yaxis().set_visible(False)
#
# axs[1, 1].imshow(reconstructions[6])
# axs[1, 1].get_xaxis().set_visible(False)
# axs[1, 1].get_yaxis().set_visible(False)
#
# axs[1, 2].imshow(reconstructions[7])
# axs[1, 2].get_xaxis().set_visible(False)
# axs[1, 2].get_yaxis().set_visible(False)
#
# axs[1, 3].imshow(reconstructions[8])
# axs[1, 3].get_xaxis().set_visible(False)
# axs[1, 3].get_yaxis().set_visible(False)
#
# axs[1, 4].set_axis_off()
#
#
# plt.savefig("Plots/9_cluster_reconstruction_new")
# plt.show()




df = pd.DataFrame(extracted_features, columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"])
df["Category"] = clusters


# print(df)

kws = dict(s=5, linewidth=0)

sns.pairplot(df, corner=True, hue="Category", plot_kws=kws, palette="colorblind")

plt.savefig("Plots/9_feature_clustering_new")
plt.show()

