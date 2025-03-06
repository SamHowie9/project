from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


np.set_printoptions(linewidth=np.inf)

encoding_dim = 7
run = 3

# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_750_epoch_32_bs_features_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_3.npy")[0]

# print(extracted_features.shape)
#
# med_features = [np.median(extracted_features.T[i]) for i in range(len(extracted_features.T))]
# print(med_features)
#
# random.seed(1)
# random_features = random.sample(range(0, extracted_features.shape[0]), 10)
#
# med_rmse = []
#
# for n_components in range(1, (extracted_features.shape[1])):
#
#     print(n_components)
#
#     pca = PCA(n_components=n_components).fit(extracted_features)
#     pca_features = pca.transform(extracted_features)
#
#     rmse = []
#
#     for i in random_features:
#
#         feature_reconstructions = pca.inverse_transform(pca_features[i])
#
#         # print(extracted_features[i])
#         # print(feature_reconstructions)
#         # print()
#
#         rmse.append(np.sqrt(np.mean(np.square(extracted_features[i] - feature_reconstructions))))
#
#     med_rmse.append(np.median(np.array(rmse)))
#
# plt.plot(med_rmse)
# plt.show()




# pca_features = pca.transform(random_features)
#
# random_features_reconstruction = pca.inverse_transform(pca_features)
#
# print(random_features)
# print(random_features_reconstruction)



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



pca = PCA(n_components=encoding_dim).fit(extracted_features)

plt.plot(range(1, encoding_dim+1), pca.explained_variance_ratio_)

plt.ylabel("Variance Explained")
plt.xlabel("Principal Components")
plt.xticks(range(1, encoding_dim+1))

print(pca.explained_variance_ratio_)


# plt.savefig("Variational Eagle/Plots/pca_scree_normalised_individually_" + str(encoding_dim) + "_features")
plt.show()



# print(pca.components_.shape)
#
# for i in range(pca.components_.shape[0]):
#     print(len(list(pca.components_[0])))