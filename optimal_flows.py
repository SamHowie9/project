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






# total_loss = np.load("Variational Eagle/Loss/Test/total_loss_beta.npy")
# reconstruction_loss = np.load("Variational Eagle/Loss/Test/reconstruction_loss_beta.npy")
# kl_loss = np.load("Variational Eagle/Loss/Test/kl_loss_beta.npy")

total_loss_original = np.load("Variational Eagle/Loss/Normalising Flow/total_loss_beta_30.npy")
reconstruction_loss_original = np.load("Variational Eagle/Loss/Normalising Flow/reconstruction_loss_beta_30.npy")
kl_loss_original = np.load("Variational Eagle/Loss/Normalising Flow/kl_loss_beta_30.npy")

total_loss_transformed = np.load("Variational Eagle/Loss/Normalising Flow/total_loss_beta_30_transformed.npy")
reconstruction_loss_transformed = np.load("Variational Eagle/Loss/Normalising Flow/reconstruction_loss_beta_30_transformed.npy")
kl_loss_transformed = np.load("Variational Eagle/Loss/Normalising Flow/kl_loss_beta_30_transformed.npy")

fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# fig, axs = plt.subplots(1, 1, figsize=(12, 5))

flow_numbers = range(1, total_loss_original.shape[0]+1)



axs[0].plot(flow_numbers, total_loss_transformed, label="Transformed")
axs[0].scatter(flow_numbers, total_loss_transformed)
axs[0].set_title("Total Loss")

axs[1].plot(flow_numbers, reconstruction_loss_transformed, label="Transformed")
axs[1].scatter(flow_numbers, reconstruction_loss_transformed)
axs[1].set_title("Reconstruction Loss")

axs[2].plot(flow_numbers, kl_loss_transformed, label="Transformed")
axs[2].scatter(flow_numbers, kl_loss_transformed)
axs[2].set_title("KL Divergence")



axs[1].plot(flow_numbers, reconstruction_loss_original, label="Original")
axs[1].scatter(flow_numbers, reconstruction_loss_original)
axs[1].set_title("Reconstruction Loss")







plt.legend()


# plt.legend()
plt.savefig("Variational Eagle/Plots/optimal_beta_30_feat", bbox_inches='tight')
plt.show()











# fig, axs = plt.subplots(3, 1, figsize=(12, 15))
#
# # fig, axs = plt.subplots(1, 1, figsize=(12, 5))
#
#
# for encoding_dim in [5, 10, 15, 20, 25, 30]:
#
#     total_loss = np.load("Variational Eagle/Loss/Test/total_loss_beta_" + str(encoding_dim) + ".npy")
#     reconstruction_loss = np.load("Variational Eagle/Loss/Test/reconstruction_loss_beta_" + str(encoding_dim) + ".npy")
#     kl_loss = np.load("Variational Eagle/Loss/Test/kl_loss_beta_" + str(encoding_dim) + ".npy")
#
#
#     # beta_words = ["0.001", "0.0001", "0.00001", "0.000001", "0.0000001"]
#     # beta_scale = [1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
#     # beta_scale = [1, 2, 2.11, 2.22, 2.33, 2.44, 2.56, 2.67, 2.78, 2.89, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
#     # beta_scale = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
#     beta_scale = [1, 2, 3, 4]
#     # beta_words = ["1e-2", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5", "5e-6", "1e-6", "5e-7", "1e-7", "5e-8", "1e-8"]
#     beta_words = ["1e-3", "1e-4", "1e-5", "1e-6"]
#
#     # axs.plot(y=reconstruction_loss, x=beta_words)
#     axs[0].plot(beta_scale, total_loss, label=encoding_dim)
#     axs[0].scatter(beta_scale, total_loss)
#     axs[0].set_title("Total Loss")
#     axs[0].set_xticks([1, 2, 3, 4])
#     axs[0].set_xticklabels(beta_words)
#
#     axs[1].plot(beta_scale, reconstruction_loss, label=encoding_dim)
#     axs[1].scatter(beta_scale, reconstruction_loss)
#     axs[1].set_title("Reconstruction Loss")
#     axs[1].set_xticks([1, 2, 3, 4])
#     axs[1].set_xticklabels(beta_words)
#
#
#     axs[2].plot(beta_scale, kl_loss, label=encoding_dim)
#     axs[2].scatter(beta_scale, kl_loss)
#     axs[2].set_title("KL Divergence")
#     axs[2].set_xticks([1, 2, 3, 4])
#     axs[2].set_xticklabels(beta_words)
#
#
# plt.legend()
# plt.savefig("Variational Eagle/Plots/optimal_beta_latent", bbox_inches='tight')
# plt.show()

