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




encoding_dim = 30
beta = 0.0001
beta_name = "0001"
epochs = 750
n_flows = 2
run = 1
batch_size = 32




# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# load the non parametric properties (restructure the dataframe to match the others)
non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
non_parametric_properties = non_parametric_properties.reset_index()
non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")

# add the non parametric properties to the other properties dataframe
all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")


# find all bad fit galaxies
bad_fit = all_properties[((all_properties["flag_r"] == 1) |
                          (all_properties["flag_r"] == 4) |
                          (all_properties["flag_r"] == 5) |
                          (all_properties["flag_r"] == 6))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# reset the index values
all_properties = all_properties.reset_index(drop=True)


print(all_properties)







# # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_" + str(run) + ".npy")[0]
#
# print(extracted_features.shape)
#
# fig, axs = plt.subplots(6, 5, figsize=(20, 24))
#
# for i in range(0, 5):
#
#     sns.histplot(x=extracted_features.T[i],ax=axs[0][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+5],ax=axs[1][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+10],ax=axs[2][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+15],ax=axs[3][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+20],ax=axs[4][i], bins=50)
#     sns.histplot(x=extracted_features.T[i+25],ax=axs[5][i], bins=50)
#     # sns.histplot(x=extracted_features.T[i+30],ax=axs[6][i], bins=50)
#
#     axs[0][i].set_xlabel("Feature " + str(i))
#     axs[1][i].set_xlabel("Feature " + str(i+5))
#     axs[2][i].set_xlabel("Feature " + str(i+10))
#     axs[3][i].set_xlabel("Feature " + str(i+15))
#     axs[4][i].set_xlabel("Feature " + str(i+20))
#     axs[5][i].set_xlabel("Feature " + str(i+25))
#     # axs[6][i].set_xlabel("Feature " + str(i+30))
#
#     for j in range(0, 5):
#         axs[j][i].set_ylabel("")
#
#
# np.set_printoptions(precision=3)
#
# print(np.log10(extracted_features.T[3]))
# print(np.format_float_positional(np.min(extracted_features.T[3]), precision=20))
# print(np.max(extracted_features.T[3]))
#
# plt.savefig("Variational Eagle/Plots/feature_distributions_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()






# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")[0]
extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T

# perform pca on the extracted features
# pca = PCA(n_components=0.999).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features = extracted_features[:len(all_properties)]
# extracted_features_switch = extracted_features.T

print(extracted_features.shape)

# rows, cols = [2, 5]
rows, cols = [6, 5]
fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))


print(rows, cols)

for i in range(rows):
    for j in range(cols):

        try:
            sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, label="Transformed")
            axs[i][j].set_xlabel("Feature " + str(j + (i*cols)))
            axs[i][j].set_ylabel("")
            axs[i][j].set_yticks([])
        except:
            print(j + (i*cols))




extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")[0]

extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T

# perform pca on the extracted features
# pca = PCA(n_components=0.999).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features = extracted_features[:len(all_properties)]
# extracted_features_switch = extracted_features.T

print(extracted_features.shape)


print(rows, cols)

for i in range(rows):
    for j in range(cols):

        try:
            sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, label="Original")
            axs[i][j].set_xlabel("Feature " + str(j + (i*cols)))
            axs[i][j].set_ylabel("")
            axs[i][j].set_yticks([])

            axs[i][j].legend()

        except:
            print(j + (i*cols))




np.set_printoptions(precision=3)

# print(np.log10(extracted_features.T[3]))
print(extracted_features.T[0])
# print(np.format_float_positional(np.min(extracted_features.T[3]), precision=20))
# print(np.max(extracted_features.T[3]))




plt.savefig("Variational Eagle/Plots/feature_distributions_planar_flows_comparison", bbox_inches='tight')
plt.show()
