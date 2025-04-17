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

# betas = [0.01, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001]
betas = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001]

# total_loss = np.load("Variational Eagle/Loss/Test/total_loss_beta.npy")
# reconstruction_loss = np.load("Variational Eagle/Loss/Test/reconstruction_loss_beta.npy")
# kl_loss = np.load("Variational Eagle/Loss/Test/kl_loss_beta.npy")

total_loss = np.load("Variational Eagle/Loss/Test/total_loss_beta_30.npy")
reconstruction_loss = np.load("Variational Eagle/Loss/Test/reconstruction_loss_beta_30.npy")
kl_loss = np.load("Variational Eagle/Loss/Test/kl_loss_beta_30.npy")

fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# fig, axs = plt.subplots(1, 1, figsize=(12, 5))


# beta_words = ["0.001", "0.0001", "0.00001", "0.000001", "0.0000001"]
# beta_scale = [1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
# beta_scale = [1, 2, 2.11, 2.22, 2.33, 2.44, 2.56, 2.67, 2.78, 2.89, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
beta_scale = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
# beta_words = ["1e-2", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5", "5e-6", "1e-6", "5e-7", "1e-7", "5e-8", "1e-8"]
beta_words = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8"]

# axs.plot(y=reconstruction_loss, x=beta_words)
axs[0].plot(beta_scale, total_loss)
axs[0].scatter(beta_scale, total_loss)
axs[0].set_title("Total Loss")
axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[0].set_xticklabels(beta_words)

axs[1].plot(beta_scale, reconstruction_loss)
axs[1].scatter(beta_scale, reconstruction_loss)
axs[1].set_title("Reconstruction Loss")
axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[1].set_xticklabels(beta_words)


axs[2].plot(beta_scale, kl_loss)
axs[2].scatter(beta_scale, kl_loss)
axs[2].set_title("KL Divergence")
axs[2].set_xticks([1, 2, 3, 4, 5, 6, 7])
axs[2].set_xticklabels(beta_words)


# plt.legend()
plt.savefig("Variational Eagle/Plots/optimal_beta", bbox_inches='tight')
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

