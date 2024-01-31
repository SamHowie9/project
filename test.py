# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd
# # from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
# # import keras
# import os
# from matplotlib import image as mpimg
# import seaborn as sns
# import random


print("Test")


#
# # set the encoding dimension (number of extracted features)
# encoding_dim = 32
#
# # # load the extracted features
# # extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")
# #
# #
# # extracted_features_switch = np.flipud(np.rot90(extracted_features))
# #
# # print(np.median(extracted_features_switch[31]))
# #
# # median_features = []
# #
# # for i in range(encoding_dim):
# #
# #     median_features.append(np.median(extracted_features_switch[i]))
# #
# #
# #
# # latent_features = []
# #
# #
# # for i in range(encoding_dim):
# #
# #     latent_images = []
# #
# #     feature_values = np.linspace(min(extracted_features_switch[i]), max(extracted_features_switch[i]), 15)
# #
# #     for j in range(15):
# #         latent_image_features = median_features
# #         latent_image_features[j] = feature_values[j]
# #         latent_images.append(latent_image_features)
# #
# #     latent_features.append(latent_images)
# #
# # latent_features = np.array(latent_features)
# #
# # print(latent_features.shape)
# # print(latent_features[0].shape)
# # print(latent_features[0][0].shape)
#
#
#
# fig, axs = plt.subplots(32, 15, figsize=(25, 10))
#
# plt.subplots_adjust(wspace=0)
#
# plt.savefig("Plots/latent_" + str(encoding_dim) + "_features")
# plt.show()
