from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

encoding_dim = 20

extracted_features = np.load("Variational Eagle/Extracted Features/Normalised to G/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]

print(extracted_features.shape)

pca = PCA(n_components=encoding_dim).fit(extracted_features.T)


plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)

plt.ylabel("Varience Explained")
plt.xlabel("Principal Components")
plt.xticks(range(1, encoding_dim+1))


plt.savefig("Variational Eagle/Plots/pca_scree_normalised_to_g_" + str(encoding_dim) + "_features")
plt.show()

# print(pca.components_.shape)
#
# for i in range(pca.components_.shape[0]):
#     print(len(list(pca.components_[0])))