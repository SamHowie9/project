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

total_loss = np.load("Variational Eagle/Loss/Test/total_loss_beta.npy")
reconstruction_loss = np.load("Variational Eagle/Loss/Test/reconstruction_loss_beta.npy")
kl_loss = np.load("Variational Eagle/Loss/Test/kl_loss_beta.npy")

fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# fig, axs = plt.subplots(1, 1, figsize=(12, 5))


beta_words = ["0.001", "0.0001", "0.00001", "0.000001", "0.0000001"]

# axs.plot(y=reconstruction_loss, x=beta_words)
axs[0].plot(beta_words, total_loss)
axs[0].scatter(beta_words, total_loss)
axs[0].set_title("Total Loss")

axs[1].plot(beta_words, reconstruction_loss)
axs[1].scatter(beta_words, reconstruction_loss)
axs[1].set_title("Reconstruction Loss")

axs[2].plot(beta_words, kl_loss)
axs[2].scatter(beta_words, kl_loss)
axs[2].set_title("KL Loss")

# plt.legend()
plt.savefig("Variational Eagle/Plots/optimal_beta", bbox_inches='tight')
plt.show()



