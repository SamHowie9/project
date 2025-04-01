import numpy as np
import pandas as pd
from PIL.GimpGradientFile import linear
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import random
from sklearn.decomposition import PCA
# from yellowbrick.cluster import KElbowVisualizer
from scipy.optimize import curve_fit
# from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
# from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

# histogram of each feature
# plot standard normal for each
# kl divergence of each feature, mean and std




encoding_dim = 25
run = 1
epochs = 750
batch_size = 32


extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]


# # standard normal distribution
# standard_normal = np.random.normal(0, 1, size=10000)
#
# fig, axs = plt.subplots(5, 5, figsize=(20, 20))
#
# for i in range(0, 5):
#
#     sns.histplot(x=extracted_features.T[i],ax=axs[0][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+5],ax=axs[1][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+10],ax=axs[2][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+15],ax=axs[3][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+20],ax=axs[4][i], bins=50)
#
#     # weights = np.ones_like(extracted_features.T[i]) * len(extracted_features.T[i])
#
#     # sns.kdeplot(x=standard_normal, ax=axs[i][0], color="black")
#     # sns.kdeplot(x=extracted_features.T[i], weights=weights, ax=axs[0][i], color="black")
#
#
#     axs[0][i].set_xlabel("Feature " + str(i))
#     axs[1][i].set_xlabel("Feature " + str(i+5))
#     axs[2][i].set_xlabel("Feature " + str(i+10))
#     axs[3][i].set_xlabel("Feature " + str(i+15))
#     axs[4][i].set_xlabel("Feature " + str(i+20))
#
#     for j in range(0, 5):
#         axs[j][i].set_ylabel("")


fig, axs = plt.subplots(1, 1, figsize=(8, 5))

sns.histplot(x=np.log10(extracted_features.T[0]), ax=axs, bins=50)
# sns.histplot(x=extracted_features.T[0], ax=axs, bins=50, kde=True, fill=False, color="black")

# standard_normal = np.random.normal(0, 1, size=10000)
# sns.histplot(x=standard_normal, ax=axs, bins=50, kde=True, fill=False, color="black")


# std_norm_x = np.linspace(-4, 4, 1000)
# std_norm_y = 1000 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (std_norm_x ** 2))
#
# plt.plot(std_norm_x, std_norm_y, color="black")


# weights = np.ones_like(extracted_features.T[0]) * len(extracted_features.T[0])

# sns.kdeplot(extracted_features.T[0], common_norm=False, ax=axs, color="black")

np.set_printoptions(precision=3)

print(np.log10(extracted_features.T[3]))
print(np.format_float_positional(np.min(extracted_features.T[3]), precision=20))
print(np.max(extracted_features.T[3]))

plt.show()