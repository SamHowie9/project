from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


np.set_printoptions(linewidth=np.inf)

encoding_dim = 24
run = 1

# 0.2113601416349411
# 0.21139435470104218
# 0.2113913595676422

# 0.21140490472316742
# 0.21135973930358887
# 0.21136081218719482

extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_750_epoch_32_bs_features_1.npy")[0]
pca = PCA(n_components=encoding_dim).fit(extracted_features)
# plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)

extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_750_epoch_32_bs_features_2.npy")[0]
pca = PCA(n_components=encoding_dim).fit(extracted_features)
# plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)

extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_750_epoch_32_bs_features_3.npy")[0]
pca = PCA(n_components=encoding_dim).fit(extracted_features)
plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)




# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_750_epoch_32_bs_features_" + str(run) + ".npy")[0]
# pca = PCA(n_components=encoding_dim).fit(extracted_features)
# plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)



plt.ylabel("Variance Explained")
plt.xlabel("Principal Components")
plt.xticks(range(1, encoding_dim+1))

print(pca.explained_variance_ratio_)


# plt.savefig("Variational Eagle/Plots/pca_scree_normalised_individually_" + str(encoding_dim) + "_features")
plt.show()
