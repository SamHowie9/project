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









fig, axs = plt.subplots(1, 2, figsize=(15, 6))


for run in [1, 2, 3, 4, 5]:

    # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_300_" + str(run) + ".npy")[0]
    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")


    print(len(extracted_features))

    pca = PCA(n_components=encoding_dim, svd_solver="full").fit(extracted_features)
    axs[0].plot(range(1, encoding_dim+1), pca.explained_variance_ratio_, label="Run "+str(run))
    axs[1].plot(range(1, 16), pca.explained_variance_ratio_[:15], label="Run "+str(run))
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.shape)


plt.ylabel("Variance Explained")
plt.xlabel("Principal Components")
axs[0].set_xticks(range(1, encoding_dim+1))
axs[1].set_xticks(range(1, 16))

axs[0].legend()
axs[1].legend()

print(pca.explained_variance_ratio_)


plt.savefig("Variational Eagle/Plots/pca_scree", bbox_inches="tight")
plt.show()
