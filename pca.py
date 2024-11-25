from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


np.set_printoptions(linewidth=np.inf)

encoding_dim = 15

extracted_features = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]

print(extracted_features.shape)

pca = PCA(n_components=encoding_dim).fit(extracted_features)
# pca_features = pca.transform(extracted_features)
#
# print(extracted_features[0])
# print(pca_features[0])
#
# new_features = pca.inverse_transform([0, 0, 0, 0, 0, 0, 0, 0])
#
# print(new_features)



# print(pca_features.shape)

# print(pca.components_.shape)

# print(extracted_features[0])
# print(pca.components_.T[0])
#
# pca_features = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
# print(pca_features.shape)
# new_features = pca.inverse_transform(pca_features.reshape(1, -1))
# print(new_features.shape)


plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)

plt.ylabel("Varience Explained")
plt.xlabel("Principal Components")
plt.xticks(range(1, encoding_dim+1))


# plt.savefig("Variational Eagle/Plots/pca_scree_normalised_individually_" + str(encoding_dim) + "_features")
plt.show()



# print(pca.components_.shape)
#
# for i in range(pca.components_.shape[0]):
#     print(len(list(pca.components_[0])))