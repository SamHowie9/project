import numpy as np
import pandas as pd
from PIL.GimpGradientFile import linear
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import random

from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
# from yellowbrick.cluster import KElbowVisualizer
from scipy.optimize import curve_fit
# from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering
# from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
plt.style.use("default")


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


np.set_printoptions(linewidth=np.inf)



run = 2
encoding_dim = 30
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32





all_properties_real = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")
all_properties_balanced = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

# print(all_properties)
# print(all_properties_balanced)

all_properties = all_properties_balanced

print(all_properties_real[all_properties_real["DiscToTotal"] < 0].shape)
print(all_properties_real[all_properties_real["DiscToTotal"] > 1].shape)


print(all_properties[all_properties["GalaxyID"] == 17917747])
print(all_properties[all_properties["GalaxyID"] == 17752121])
print(all_properties[all_properties["GalaxyID"] == 9526568])
print(all_properties.iloc[475])




# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")
#
# # load the non parametric properties (restructure the dataframe to match the others)
# non_parametric_properties = pd.read_hdf("Galaxy Properties/Eagle Properties/Ref100N1504.hdf5", key="galface/r")
# non_parametric_properties = non_parametric_properties.reset_index()
# non_parametric_properties = non_parametric_properties.sort_values(by="GalaxyID")
#
# # load the disk-total ratios
# disk_total = pd.read_csv("Galaxy Properties/Eagle Properties/disk_to_total.csv", comment="#")
#
# print(disk_total)
#
# # add the non parametric properties, and the disk-total to the other properties dataframe
# all_properties = pd.merge(all_properties, non_parametric_properties, on="GalaxyID")
# all_properties = pd.merge(all_properties, disk_total, on="GalaxyID")


# # find all bad fit galaxies
# bad_fit = all_properties[((all_properties["flag_r"] == 1) |
#                           (all_properties["flag_r"] == 4) |
#                           (all_properties["flag_r"] == 5) |
#                           (all_properties["flag_r"] == 6))].index.tolist()
#
# # remove those galaxies
# for galaxy in bad_fit:
#     all_properties = all_properties.drop(galaxy, axis=0)
#
# # reset the index values
# all_properties = all_properties.reset_index(drop=True)




# print(all_properties)
#
# print(all_properties.sort_values(by="MassType_Star"))
#
# print(all_properties[all_properties["flag"] != 0])
# print(all_properties[all_properties["flag_sersic"] != 0])










# original dataset

# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Original/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/PCA/pca_features_" + str(encoding_dim) + "_features.npy")
# encoding_dim = extracted_features.shape[1]
#
#
# # perform pca on the extracted features
# pca = PCA(n_components=13).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
#
#
# # account for the training data in the dataframe
# # all_properties = all_properties.drop(all_properties.tail(200).index, inplace=True)
# all_properties = all_properties.iloc[:-200]
#
#
# print(all_properties)
# print()








# print(all_properties.sort_values(by="re_r"))





# fully balanced dataset

# for encoding_dim in range(5, 21):

# account for the testing dataset
# all_properties = all_properties.iloc[:-200]


# for n_flows in [n_flows]:
# for run in [1, 2, 3]:
for run in [1, 2, 3, 4, 5]:

    print(n_flows, run)

    # load the extracted features
    # extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
    # extracted_features = np.load("Variational Eagle/Extracted Features/Test/bce_beta_01.npy")[0]
    # extracted_features = np.load("Variational Eagle/Extracted Features/Final/bce_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_300_" + str(run) + ".npy")[0]
    # extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default.npy")[0]
    extracted_features = np.load("Variational Eagle/Extracted Features/Normalising Flow Balanced/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
    encoding_dim = extracted_features.shape[1]

    # print(extracted_features.shape)

    # remove augmented images
    # extracted_features = extracted_features[:len(all_properties)]

    # perform pca on the extracted features
    pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
    extracted_features = pca.transform(extracted_features)
    # extracted_features = extracted_features[:len(all_properties)]

    print(pca.explained_variance_ratio_)





    # selected_indices = all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index.tolist()
    # all_properties = all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")]

    # selected_indices = all_properties[all_properties["DiscToTotal"] >= 0.5].index.tolist()
    # all_properties = all_properties[all_properties["DiscToTotal"] >= 0.5]
    #
    #
    # print(all_properties)
    # extracted_features = np.array([extracted_features[i] for i in selected_indices])
    # all_properties = all_properties.reset_index(drop=True)
    # print(extracted_features.shape)
    # print(selected_indices)





    # spirals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] > 0.2].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] > 0.2]



    # ellipticals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] < 0.1].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] < 0.1]


    # transitional
    # spiral_indices = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")].index.tolist()
    # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")]


    # original images
    # all_properties = all_properties_real
    # extracted_features = extracted_features[:len(all_properties)]



    # print(extracted_features.shape)
    #
    # print(all_properties)


    # correlation plot

    # dataframe to contain correlations between each feature and each property
    # correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
    correlation_df = pd.DataFrame(columns=list(all_properties.columns)[1:])


    # loop through each extracted feature
    for feature in range(0, len(extracted_features.T)):

        # create a list to contain the correlation between that feature and each property
        correlation_list = []

        # loop through each property
        for gal_property in range(1, len(all_properties.columns)):

            # skip the flag property
            # if gal_property == 6:
            #     continue

            # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
            correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties.iloc[:, gal_property])[0][1]
            correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]


            # add the strongest type of correlation
            correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

        # add all the correlations for that feature to the dataframe
        correlation_df.loc[len(correlation_df)] = correlation_list



    # print(correlation_df)

    # correlation_df = correlation_df.iloc[[12, 21, 27]]

    # set the figure size
    # plt.figure(figsize=(20, extracted_features.T.shape[0]))
    plt.figure(figsize=(35, correlation_df.shape[0]))


    # properties to plot
    # selected_properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]
    selected_properties = ["n_r", "DiscToTotal", "re_r", "rhalf_ellip", "pa_r", "q_r",  "mag_r", "MassType_Star", "InitialMassWeightedStellarAge", "StarFormationRate", "gini", "m20", "concentration", "asymmetry", "smoothness"]

    # plot a heatmap for the dataframe (with annotations)
    # ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
    ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={'label': 'Correlation'})



    plt.yticks(rotation=0)
    plt.ylabel("Extracted Features", fontsize=15)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    ax.figure.axes[-1].yaxis.label.set_size(15)



    def wrap_labels(ax, width, break_long_words=False):

        labels = []
        # for label in ax.get_xticklabels():
            # text = label.get_text()

        label_names = ["Sersic Index", "Disk-Total Ratio", "Semi - Major Axis", "Half Light Radius", "Position Angle", "Axis Ratio", "AB Magnitude", "Stellar Mass", "Stellar Age", "Star Formation Rate", "Gini Coefficient", "M20", "Concentration", "Asymmetry", "Smoothness"]

        for text in label_names:
            labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0, fontsize=15)

    wrap_labels(ax, 10)



    # plt.savefig("Variational Eagle/Correlation Plots/fully_balanced_" + str(encoding_dim) + "_feature_vae_all_property_correlation_" + str(run), bbox_inches='tight')
    # plt.savefig("Variational Eagle/Correlation Plots/Correlation Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_correlation_" + str(run), bbox_inches='tight')
    # plt.savefig("Variational Eagle/Correlation Plots/Final/top_4_pca_" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_correlation_" + str(run), bbox_inches='tight')
    plt.savefig("Variational Eagle/Correlation Plots/Normalising Flows Balanced/PCA/latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced", bbox_inches='tight')
    plt.show()







# scatter plot

# # properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]
# properties = ["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]
#
#
# all_properties = all_properties[["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]]
#
# # print(all_properties)
#
# property_labels = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]
#
# # fig, axs = plt.subplots(encoding_dim, len(properties), figsize=(40, (encoding_dim * 4)))
# fig, axs = plt.subplots(extracted_features.T.shape[0], len(properties), figsize=(40, (encoding_dim * 4)))
#
# # fig, axs = plt.subplots(19, len(properties), figsize=(40, 70))
#
#
# for i, property in enumerate(properties):
#
#     axs[0][i].set_title(property_labels[i], fontsize=20)
#
#     # for feature in range(0, 19):
#     # for feature in range(0, encoding_dim):
#     for feature in range(0, extracted_features.T.shape[0]):
#
#         axs[feature][i].scatter(x=extracted_features.T[feature], y=all_properties[property], s=0.5)
#         # axs[feature][i].scatter(x=extracted_features.T[feature+19], y=all_properties[property], s=0.5)
#
#         # sns.kdeplot(data=all_properties, x=extracted_features.T[feature], y=all_properties[property], gridsize=200)
#
#         axs[feature][i].set_xlabel("Feature " + str(feature), fontsize=12)
#         # axs[feature][i].set_xlabel("Feature " + str(feature+19), fontsize=12)
#         axs[feature][i].set_ylabel(None)
#         axs[feature][i].tick_params(labelsize=12)
#
# plt.savefig("Variational Eagle/Correlation Plots/scatter_fully_balanced_" + str(encoding_dim) + "_feature_vae_all_property_correlation_" + str(run), bbox_inches='tight')
# # plt.savefig("Variational Eagle/Correlation Plots/scatter_" + str(encoding_dim) + "_feature_all_property_correlation_p2", bbox_inches='tight')
# plt.show()



















def quadratic(a, b, c, x):
    return (a * x * x) + (b * x) + c

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c




# def density_scatter(x ,y, axs, sort=True, bins=20, **kwargs):
#
#     # find the density colour based on the histogram
#     data ,x_e, y_e = np.histogram2d(x, y, bins=bins, density=True )
#     # z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:] + y_e[:-1])), data ,np.vstack([x,y]).T, method="splinef2d", bounds_error=False)
#     z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:] + y_e[:-1])), data ,np.vstack([x,y]).T, method="nearest", bounds_error=False)
#
#     # replace nan with 0 (where there are no points)
#     z[np.where(np.isnan(z))] = 0.0
#
#     # sort the points by density, so that the densest points are plotted last
#     if sort:
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
#
#     # make and return the scatter plot with the colour corresponding to the density
#     axs.scatter(x, y, c=z, **kwargs )
#     return axs


def density_scatter(x ,y, axs, **kwargs):

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axs.scatter(x, y, c=z, **kwargs)
    return axs



# fig, axs = plt.subplots(1, 1, figsize=(12, 10))
#
# density_scatter(all_properties["n_r"], all_properties["DiscToTotal"], axs=axs, s=5)
# axs.set_xlabel("Sersic Index")
# axs.set_ylabel("Disk-Total Ratio")
#
# plt.show()
#
#
# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
#
# density_scatter(extracted_features.T[1], all_properties["DiscToTotal"], axs=axs[0], s=5)
# axs[0].set_xlabel("Feature 1")
# axs[0].set_ylabel("Disk-Total Ratio")
#
# density_scatter(extracted_features.T[3], all_properties["DiscToTotal"], axs=axs[1], s=5)
# axs[1].set_xlabel("Feature 3")
# axs[1].set_ylabel("Disk-Total Ratio")
#
# density_scatter(extracted_features.T[6], all_properties["DiscToTotal"], axs=axs[2], s=5)
# axs[2].set_xlabel("Feature 6")
# axs[2].set_ylabel("Disk-Total Ratio")
#
#
# plt.show()



fig, axs = plt.subplots(1, 1, figsize=(8, 7))

sns.histplot(x=all_properties["DiscToTotal"], ax=axs, bins=50)
axs.set_xlabel("Disk-Total Ratio")

plt.savefig("Variational Eagle/Plots/disk_total_distribution")
plt.show()






# structure measurement comparison

# fig, axs = plt.subplots(1, 5, figsize=(25, 5))
#
# density_scatter(extracted_features.T[3], all_properties["n_r"], axs=axs[0], s=2)
# axs[0].set_title("Sersic Index")
# axs[0].set_xlabel("Feature 3")
# axs[0].set_ylabel("Sersic Index")
#
# density_scatter(extracted_features.T[3], all_properties["gini"], axs=axs[1], s=2)
# axs[1].set_title("Gini Coefficient")
# axs[1].set_xlabel("Feature 3")
# axs[1].set_ylabel("Gini Coefficient")
#
# density_scatter(extracted_features.T[3], all_properties["concentration"], axs=axs[2], s=2)
# axs[2].set_title("Concentration")
# axs[2].set_xlabel("Feature 3")
# axs[2].set_ylabel("Concentration")
#
# density_scatter(extracted_features.T[0], all_properties["asymmetry"], axs=axs[3], s=2)
# axs[3].set_title("Asymmetry")
# axs[3].set_xlabel("Feature 0")
# axs[3].set_ylabel("Asymmetry")
#
# density_scatter(extracted_features.T[0], all_properties["smoothness"], axs=axs[4], s=2)
# axs[4].set_title("Smoothness")
# axs[4].set_xlabel("Feature 0")
# axs[4].set_ylabel("Smoothness")
#
#
# plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_30_1", bbox_inches='tight')
# plt.show()





# fig, axs = plt.subplots(2, 4, figsize=(25, 10))
#
# density_scatter(extracted_features.T[4], all_properties["n_r"], axs=axs[0][0], s=10)
# axs[0][0].set_title("Sersic Index")
# axs[0][0].set_xlabel("Feature 4")
# axs[0][0].set_ylabel("Sersic Index")
# # axs[0][0].set_xlim(-4, 4)
# # axs[0][0].set_ylim(0, 6)
#
# density_scatter(extracted_features.T[2], abs(all_properties["pa_r"]), axs=axs[0][1], s=10)
# axs[0][1].set_title("Position Angle")
# axs[0][1].set_xlabel("Feature 2")
# axs[0][1].set_ylabel("Position Angle (°)")
# # axs[0][1].set_yticks([0, 45, 90])
# # axs[0][1].set_xlim(-3.5, 3.5)
#
# density_scatter(extracted_features.T[2], all_properties["q_r"], axs=axs[0][2], s=10)
# axs[0][2].set_title("Axis Ratio")
# axs[0][2].set_xlabel("Feature 2")
# axs[0][2].set_ylabel("Axis Ratio")
# # axs[0][2].set_xlim(-3, 3)
#
# density_scatter(extracted_features.T[3], all_properties["m20"], axs=axs[0][3], s=10)
# axs[0][3].set_title("M20")
# axs[0][3].set_xlabel("Feature 0")
# axs[0][3].set_ylabel("M20")
# # axs[0][3].set_xlim(-6, 4)
# # axs[0][3].set_ylim(-2.5, -1.2)
#
# density_scatter(extracted_features.T[4], all_properties["gini"], axs=axs[1][0], s=10)
# axs[1][0].set_title("Gini Coefficient")
# axs[1][0].set_xlabel("Feature 4")
# axs[1][0].set_ylabel("Gini Coefficient")
# # axs[1][0].set_xlim(-4, 4)
# # axs[1][0].set_ylim(0.4, 0.65)
#
# density_scatter(extracted_features.T[4], all_properties["concentration"], axs=axs[1][1], s=10)
# axs[1][1].set_title("Concentration")
# axs[1][1].set_xlabel("Feature 4")
# axs[1][1].set_ylabel("Concentration")
# # axs[1][1].set_xlim(-3, 3)
# # axs[1][1].set_ylim(2, 5)
#
# density_scatter(extracted_features.T[0], abs(all_properties["asymmetry"]), axs=axs[1][2], s=10)
# axs[1][2].set_title("Asymmetry")
# axs[1][2].set_xlabel("Feature 0")
# axs[1][2].set_ylabel("Asymmetry")
# # axs[1][2].set_xlim(-4, 4)
# # axs[1][2].set_ylim(0, 0.5)
#
# density_scatter(extracted_features.T[0], abs(all_properties["smoothness"]), axs=axs[1][3], s=10)
# axs[1][3].set_title("Smoothness")
# axs[1][3].set_xlabel("Feature 0")
# axs[1][3].set_ylabel("Smoothness")
# # axs[1][3].set_xlim(-4, 4)
# # axs[1][3].set_ylim(0, 0.1)
#
#
# plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_" + str(encoding_dim) + "_" + str(n_flows) + "_" + str(run) + "_unknown", bbox_inches='tight')
#
# plt.show()





# fig, axs = plt.subplots(3, 8, figsize=(50, 15))
#
# for i, feature in enumerate([12, 21, 27]):
#
#     density_scatter(extracted_features.T[feature], all_properties["n_r"], axs=axs[i][0], s=10)
#     axs[i][0].set_title("Sersic Index")
#     axs[i][0].set_xlabel("Feature " + str(feature))
#     axs[i][0].set_ylabel("Sersic Index")
#     # axs[0][0].set_xlim(-4, 4)
#     # axs[0][0].set_ylim(0, 6)
#
#     density_scatter(extracted_features.T[feature], abs(all_properties["pa_r"]), axs=axs[i][1], s=10)
#     axs[i][1].set_title("Position Angle")
#     axs[i][1].set_xlabel("Feature " + str(feature))
#     axs[i][1].set_ylabel("Position Angle (°)")
#     # axs[0][1].set_yticks([0, 45, 90])
#     # axs[0][1].set_xlim(-3.5, 3.5)
#
#     density_scatter(extracted_features.T[feature], all_properties["q_r"], axs=axs[i][2], s=10)
#     axs[i][2].set_title("Axis Ratio")
#     axs[i][2].set_xlabel("Feature " + str(feature))
#     axs[i][2].set_ylabel("Axis Ratio")
#     # axs[0][2].set_xlim(-3, 3)
#
#     density_scatter(extracted_features.T[feature], all_properties["m20"], axs=axs[i][3], s=10)
#     axs[i][3].set_title("M20")
#     axs[i][3].set_xlabel("Feature " + str(feature))
#     axs[i][3].set_ylabel("M20")
#     # axs[0][3].set_xlim(-6, 4)
#     # axs[0][3].set_ylim(-2.5, -1.2)
#
#     density_scatter(extracted_features.T[feature], all_properties["gini"], axs=axs[i][4], s=10)
#     axs[i][4].set_title("Gini Coefficient")
#     axs[i][4].set_xlabel("Feature " + str(feature))
#     axs[i][4].set_ylabel("Gini Coefficient")
#     # axs[1][0].set_xlim(-4, 4)
#     # axs[1][0].set_ylim(0.4, 0.65)
#
#     density_scatter(extracted_features.T[feature], all_properties["concentration"], axs=axs[i][5], s=10)
#     axs[i][5].set_title("Concentration")
#     axs[i][5].set_xlabel("Feature " + str(feature))
#     axs[i][5].set_ylabel("Concentration")
#     # axs[1][1].set_xlim(-3, 3)
#     # axs[1][1].set_ylim(2, 5)
#
#     density_scatter(extracted_features.T[feature], abs(all_properties["asymmetry"]), axs=axs[i][6], s=10)
#     axs[i][6].set_title("Asymmetry")
#     axs[i][6].set_xlabel("Feature " + str(feature))
#     axs[i][6].set_ylabel("Asymmetry")
#     # axs[1][2].set_xlim(-4, 4)
#     # axs[1][2].set_ylim(0, 0.5)
#
#     density_scatter(extracted_features.T[feature], abs(all_properties["smoothness"]), axs=axs[i][7], s=10)
#     axs[i][7].set_title("Smoothness")
#     axs[i][7].set_xlabel("Feature " + str(feature))
#     axs[i][7].set_ylabel("Smoothness")
#     # axs[1][3].set_xlim(-4, 4)
#     # axs[1][3].set_ylim(0, 0.1)
#
#
# plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_subset_transformed", bbox_inches='tight')
# plt.show()






# physical property comparison

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
#
# density_scatter(extracted_features.T[0], all_properties["re_r"], axs=axs[0], s=2)
#
# density_scatter(extracted_features.T[0], all_properties["mag_r"], axs=axs[1], s=2)
#
# density_scatter(extracted_features.T[0], all_properties["MassType_Star"], axs=axs[2], s=2)
#
# density_scatter(extracted_features.T[0], all_properties["StarFormationRate"], axs=axs[2], s=2)
#
#
#
# plt.savefig("Variational Eagle/Plots/physical_property_comparisons_30_1", bbox_inches='tight')
# plt.show()











# combinations of features

# linear_model = LinearRegression()
# linear_model.fit(abs(extracted_features), list(all_properties["q_r"]))
#
# predicted_feature = linear_model.predict(abs(extracted_features))
#
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# # axs.set_ylim(0, 2.5e11)
#
# # plt.scatter(predicted_feature, list(all_properties["q_r"]), s=2)
# density_scatter(x=predicted_feature, y=all_properties["q_r"], axs=axs, s=5)
#
# axs.set_ylabel("Actual Ratio")
# axs.set_xlabel("Predicted Ratio")
#
# plt.savefig("Variational Eagle/Plots/q")
# plt.show()








# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# density_scatter(x=extracted_features.T[2], y=abs(all_properties["q_r"]), axs=axs, s=5)
# plt.savefig("Variational Eagle/Plots/q")
# plt.show()





