from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


run = 3
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 300
batch_size = 32




np.set_printoptions(linewidth=np.inf)









fig, axs = plt.subplots(1, 2, figsize=(20, 8))


for run in [1, 2, 3]:

    # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_300_" + str(run) + ".npy")[0]
    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")


    print(len(extracted_features))

    pca = PCA(n_components=encoding_dim).fit(extracted_features)
    axs[0].plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)
    axs[1].plot(range(5, encoding_dim+1), pca.explained_variance_ratio_[4:])
    print(pca.explained_variance_ratio_)


plt.ylabel("Variance Explained")
plt.xlabel("Principal Components")
axs[0].set_xticks(range(1, encoding_dim+1))
axs[1].set_xticks(range(5, encoding_dim+1))

print(pca.explained_variance_ratio_)


# plt.savefig("Variational Eagle/Plots/pca_scree", bbox_inches="tight")
plt.show()
