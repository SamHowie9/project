import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import random

from matplotlib.pyplot import figure
from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



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


particle_count = pd.read_csv("Galaxy Properties/Eagle Properties/particle_counts.csv", comment="#")

zero = particle_count[particle_count["Count_Dust"] == 0]
small = particle_count[particle_count["Count_Dust"].between(0, 250, inclusive="neither")]
large = particle_count[particle_count["Count_Dust"] > 250]


ab_mags_dusty = pd.read_csv("Galaxy Properties/Eagle Properties/ab_magnitudes_dusty.txt", comment="#")
ab_mags = pd.read_csv("Galaxy Properties/Eagle Properties/ab_magnitudes.csv", comment="#")


ab = pd.merge(ab_mags_dusty, ab_mags, on="GalaxyID")
# print(ab[ab["GalaxyID"] == 14289611])
subset = ab[ab["GalaxyID"].isin([14289611, 16090542, 2637010, 6643600, 19195804])]
print(subset)
subset.columns = ["GalaxyID", "dusty g", "dusty r", "dusty i", "no dust g", "no dust r", "no dust i"]
print(subset[["GalaxyID", "dusty g", "no dust g", "dusty i", "no dust i", "dusty r", "no dust r"]])



# print(particle_count)
# print(zero)
# print(small)
print(large.sort_values(by="Count_Dust"))

# print(particle_count)
#
# bins = [0, 10, 100, 200, 300, 400, 500, 1000]
#
# # Plot
# counts, edges, patches = plt.hist(particle_count["Count_Dust"], bins=bins, edgecolor='black')
#
# # Set x-ticks with last bin labeled "1000+"
# # xticks = list(range(0, 1001, 100)) + [1100]
# # xticklabels = [str(x) for x in range(0, 1001, 100)] + ['1000+']
# # plt.xticks(xticks, xticklabels)
#
# plt.xlabel('Galaxy Size')
# plt.ylabel('Count')
# plt.title('Galaxy Size Distribution')
# plt.show()