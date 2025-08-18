import numpy as np
import pandas as pd
from PIL.GimpGradientFile import linear
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
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
from scipy.stats import norm





# histogram of each feature
# plot standard normal for each
# kl divergence of each feature, mean and std




run = 3
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32




for run in [2, 5, 7, 10, 12, 15, 17, 18, 19, 20, 22, 23]:

    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

    extracted_features_switch = extracted_features.T

    # perform pca on the extracted features
    # pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
    # extracted_features = pca.transform(extracted_features)
    # extracted_features_switch = extracted_features.T

    print(extracted_features.shape)

    # rows, cols = [2, 5]
    # rows, cols = [6, 5]
    rows, cols = [4, 8]
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

    # standard_normal = np.random.normal(loc=0, scale=1, size=extracted_features.shape[0])
    standard_normal = np.random.normal(loc=0, scale=1, size=100000)

    print(rows, cols)

    for i in range(rows):
        for j in range(cols):

            try:
                # sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, element="poly", label="Transformed")
                sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, stat="density", edgecolor="none", linewidth=0, color="grey")

                for patch in axs[i][j].patches:
                    patch.set_rasterized(True)

                # if (j + (i*cols) + 1) in [6, 8, 10, 11, 12, 13, 15, 19, 27, 30]:
                #     sns.histplot(x=standard_normal,ax=axs[i][j], kde=True, stat="density", color="black", fill=False, alpha=0, bins=50)

                # sns.histplot(x=standard_normal,ax=axs[i][j], kde=True, stat="density", color="black", fill=False, alpha=0, bins=50)

                mean = np.mean(extracted_features.T[j + (i*cols)])
                std = np.std(extracted_features.T[j + (i*cols)])

                approx_dist_x = np.linspace(mean-(4*std), mean+(4*std), 1000)
                approx_dist_y = norm.pdf(approx_dist_x, mean, std)
                axs[i][j].plot(approx_dist_x, approx_dist_y, color="black")

                # approx_dist = np.random.normal(loc=mean, scale=std, size=1000000)
                # sns.histplot(x=approx_dist,ax=axs[i][j], kde=True, stat="density", color="black", fill=False, alpha=0, bins=50)


                axs[i][j].set_title("(" + str(round(mean, 3)) + ", " + str(round(std, 3)) + ")", fontsize=30)
                axs[i][j].set_xlabel("Feature " + str(j + (i*cols) + 1), fontsize=30)


                axs[i][j].set_ylabel("")
                axs[i][j].set_yticks([])

                ticks = [round(mean-3*std, 3), round(mean, 3), round(mean+3*std, 3)]
                axs[i][j].set_xticks(ticks)
                axs[i][j].tick_params(labelsize=22)

                axs[i][j].set_xlim(mean-(4*std), mean+(4*std))

            except:
                print(j + (i*cols))

    axs[3][6].set_axis_off()
    axs[3][7].set_axis_off()

    fig.subplots_adjust(wspace=0.1, hspace=0.6)

    plt.savefig("Variational Eagle/Distribution Plots/Latent/latent_feature_distributions_" + str(run) , bbox_inches='tight')
    # plt.savefig("Variational Eagle/Plots/latent_feature_distributions.pdf", bbox_inches='tight')
    plt.show(block=False)
    plt.close()







# rows, cols = [6, 10]
# fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
#
# # standard_normal = np.random.normal(loc=0, scale=1, size=extracted_features.shape[0])
# standard_normal = np.random.normal(loc=0, scale=1, size=100000)
#
# print(rows, cols)
#
# # for i in range(rows):
# #     for j in range(int(cols/2)):
# #
# #         index = j + int(i*cols/2)
# #
# #         try:
# #             # sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, element="poly", label="Transformed")
# #             sns.histplot(x=extracted_features.T[index],ax=axs[i][j], bins=50, stat="density", label="Transformed")
# #             # sns.histplot(x=standard_normal,ax=axs[i][j], kde=True, stat="density", color="black", fill=False, alpha=0, bins=50)
# #
# #             axs[i][j].set_xlabel("Feature " + str(index))
# #             axs[i][j].set_ylabel("")
# #             axs[i][j].set_yticks([])
# #         except:
# #             print(index)
#
# for i in range(rows):
#     for j in range(int(cols/2)):
#
#         index = j + int(i*cols/2)
#
#         try:
#             # sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, element="poly", label="Transformed")
#             sns.histplot(x=extracted_features.T[index],ax=axs[i][j+5], bins=50, stat="density", label="Transformed")
#             sns.histplot(x=standard_normal,ax=axs[i][j+5], kde=True, stat="density", color="black", fill=False, alpha=0, bins=50)
#
#             axs[i][j+5].set_xlabel("Feature " + str(index))
#             axs[i][j+5].set_ylabel("")
#             axs[i][j+5].set_yticks([])
#         except:
#             print(index)
#
#
# np.set_printoptions(precision=3)
#
#
# plt.savefig("Variational Eagle/Plots/feature_distributions_latent", bbox_inches='tight')
# plt.show()











#
# # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced Mean/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
#
# # extracted_features = extracted_features[:len(all_properties)]
# extracted_features_switch = extracted_features.T
#
# # perform pca on the extracted features
# # pca = PCA(n_components=0.999).fit(extracted_features)
# # extracted_features = pca.transform(extracted_features)
# # extracted_features = extracted_features[:len(all_properties)]
# # extracted_features_switch = extracted_features.T
#
# print(extracted_features.shape)
#
# # rows, cols = [2, 5]
# rows, cols = [6, 5]
# fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
#
#
# print(rows, cols)
#
# for i in range(rows):
#     for j in range(cols):
#
#         try:
#             # sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, element="poly", label="Transformed")
#             sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, label="Transformed")
#             axs[i][j].set_xlabel("Feature " + str(j + (i*cols)))
#             axs[i][j].set_ylabel("")
#             axs[i][j].set_yticks([])
#         except:
#             print(j + (i*cols))
#
#
#
#
# extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")
#
# # extracted_features = extracted_features[:len(all_properties)]
# extracted_features_switch = extracted_features.T
#
# # perform pca on the extracted features
# # pca = PCA(n_components=0.999).fit(extracted_features)
# # extracted_features = pca.transform(extracted_features)
# # extracted_features = extracted_features[:len(all_properties)]
# # extracted_features_switch = extracted_features.T
#
#
# standard_normal = np.random.normal(loc=0, scale=1, size=extracted_features.shape[0])
# print(len(standard_normal))
#
#
# for i in range(rows):
#     for j in range(cols):
#
#         try:
#             sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], kde=True, color="C1", fill=False, alpha=0, bins=50)
#             # sns.histplot(x=extracted_features.T[j + (i*cols)],ax=axs[i][j], bins=50, label="Original")
#             axs[i][j].set_xlabel("Feature " + str(j + (i*cols)))
#             axs[i][j].set_ylabel("")
#             axs[i][j].set_yticks([])
#
#
#             # sns.histplot(x=standard_normal,ax=axs[i][j], kde=True, color="black", fill=False, alpha=0, bins=50)
#
#
#             orange_line = mlines.Line2D([], [], color="C1", linestyle='-', label="Original")
#             black_line = mlines.Line2D([], [], color="black", linestyle='-', label="Gaussian")
#             handles = axs[i][j].get_legend_handles_labels()[0] + [orange_line, black_line]
#             axs[i][j].legend(handles=handles)
#
#             # axs[i][j].legend()
#
#         except:
#             print(j + (i*cols))
#
#
#
#
# np.set_printoptions(precision=3)
#
# # print(np.log10(extracted_features.T[3]))
# print(extracted_features.T[0])
# # print(np.format_float_positional(np.min(extracted_features.T[3]), precision=20))
# # print(np.max(extracted_features.T[3]))
#
#
#
#
# plt.savefig("Variational Eagle/Distribution Plots/Normal/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_all_gaussian", bbox_inches='tight')
# plt.show()
