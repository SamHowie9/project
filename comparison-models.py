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



for run in range(1, 26):

    # balanced_loss = np.load("Variational Eagle/Loss/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    balanced_loss = np.load("Variational Eagle/Loss/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_individual_residual.npy")
    balanced_loss = np.mean(balanced_loss)
    balanced_losses.append(balanced_loss[1])

# for run in range(1, 11):

    spiral_loss = np.load("Variational Eagle/Loss/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    # spiral_loss = np.load("Variational Eagle/Loss/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_individual_residual.npy")
    spiral_loss = np.mean(spiral_loss)
    spiral_losses.append(spiral_loss[1])

    transitional_loss = np.load("Variational Eagle/Loss/Transitional/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    # transitional_loss = np.load("Variational Eagle/Loss/Transitional/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_individual_residual.npy")
    transitional_loss = np.mean(transitional_loss)
    transitional_losses.append(transitional_loss[1])

    # elliptical_loss = np.load("Variational Eagle/Loss/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
    elliptical_loss = np.load("Variational Eagle/Loss/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_individual_residual.npy")
    elliptical_loss = np.mean(elliptical_loss)
    elliptical_losses.append(elliptical_loss[1])


all_losses = [balanced_losses, spiral_losses, transitional_losses, elliptical_losses]




# fig, axs = plt.subplots(1, 1, figsize=(20, 20))
#
# axs.boxplot(x=all_losses)
#
# plt.show()






# fig, axs = plt.subplots(1, 1, figsize=(20, 20))
#
# axs.scatter([0]*len(balanced_losses), balanced_losses)
# axs.scatter([1]*len(spiral_losses), spiral_losses)
# axs.scatter([2]*len(transitional_losses), transitional_losses)
# axs.scatter([3]*len(elliptical_losses), elliptical_losses)
#
# plt.show()





mean_losses = [np.mean(balanced_losses), np.mean(spiral_losses), np.mean(transitional_losses), np.mean(elliptical_losses)]
print(mean_losses)

fig, axs = plt.subplots(1, 1, figsize=(5, 10))

axs.scatter([0, 0, 0, 0], mean_losses, s=150)

# labels = ["Balanced", "Disk-Dominated", "Transitional", "Bulge-Dominated"]

axs.text(0.05, mean_losses[0], "Balanced", fontsize=20, ha="left", va="center")
axs.text(0.05, mean_losses[1], "Disk-Dominated", fontsize=20, ha="left", va="center")
axs.text(0.05, mean_losses[2], "Transitional", fontsize=20, ha="left", va="center")
axs.text(0.05, mean_losses[3], "Bulge-Dominated", fontsize=20, ha="left", va="center")

axs.set_ylabel("Reconstruction Loss (BCE)", fontsize=20, labelpad=20)
axs.tick_params(labelsize=20)
axs.set_xlim(-0.1, 0.6)
axs.set_xticks([])
# axs.set_yticks([round(mean_losses[2], 3), round(mean_losses[1], 3), round(mean_losses[0], 3), round(mean_losses[3], 3)])
# axs.set_yticks([0.204, 0.206, 0.208, 0.210, 0.212, 0.214, 0.216])

plt.savefig("Variational Eagle/Plots/subset_loss_comparison_2", bbox_inches="tight")
plt.show()





# fig, axs = plt.subplots(1, 4, figsize=(20, 20))
#
# axs[0].boxplot(balanced_losses)
# axs[1].boxplot(spiral_losses)
# axs[2].boxplot(transitional_losses)
# axs[3].boxplot(elliptical_losses)
#
# plt.show()
