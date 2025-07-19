import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random
from sklearn.decomposition import PCA



# pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)



# run = 16
# encoding_dim = 30
# n_flows = 0
# beta = 0.0001
# beta_name = "0001"
# epochs = 750
# batch_size = 32
#
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_750_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
# print(extracted_features.shape)
#
# pca = PCA(n_components=encoding_dim, svd_solver="full").fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# variances = pca.explained_variance_ratio_
#
# df = pd.DataFrame(columns=["Principal Component", "Standard Deviation", "Variance Explanation"])
#
# for i in range(extracted_features.shape[1]):
#
#     feature = extracted_features.T[i]
#     std = np.std(feature)
#     # print(std, variances[i])
#
#     df.loc[len(df)] = [i+1, std, variances[i]]
#
# print(df)


A = []
B = A + [1]
B = B + [2, 3]

print(B)