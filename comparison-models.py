import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



run = 16
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32



balanced_losses = []
spiral_losses = []
elliptical_losses = []
transitional_losses = []



# for run in range(1, 26):
#     balanced_loss = np.load("Variational Eagle/Loss/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
#     balanced_losses.append(balanced_loss)

for run in range(1, 11):
    spiral_loss = np.load("Variational Eagle/Loss/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    spiral_losses.append(spiral_loss[1])

    transitional_loss = np.load("Variational Eagle/Loss/Transitional/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    transitional_losses.append(transitional_loss[1])

    elliptical_loss = np.load("Variational Eagle/Loss/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    elliptical_losses.append(elliptical_loss[1])

all_losses = [spiral_losses, transitional_losses, elliptical_losses]

fig, axs = plt.subplots(1, 1, figsize=(20, 20))

# axs.boxplot(balanced_losses)
axs.boxplot(spiral_losses)
axs.boxplot(transitional_losses)
axs.boxplot(elliptical_losses)

# axs.boxplot(x=all_losses)

plt.show()