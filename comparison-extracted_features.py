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

plt.style.use("default")


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


encoding_dim = 15
run = 3
epochs = 750
batch_size = 32


# for encoding_dim in [5, 6, 7, 8, 10]:
#
# for run in [1, 2, 3]:


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
bad_fit = all_properties[((all_properties["flag_r"] == 4) | (all_properties["flag_r"] == 1) | (all_properties["flag_r"] == 5))].index.tolist()

# remove those galaxies
for galaxy in bad_fit:
    all_properties = all_properties.drop(galaxy, axis=0)

# reset the index values
all_properties = all_properties.reset_index(drop=True)


# print(all_properties)
print(all_properties[all_properties["n_r"] >=4].sort_values("re_r"))












# original dataset

# # load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Original/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# # extracted_features = np.load("Variational Eagle/Extracted Features/PCA/pca_features_" + str(encoding_dim) + "_features.npy")
# encoding_dim = extracted_features.shape[1]
# extracted_features_switch = extracted_features.T
#
#
# # perform pca on the extracted features
# pca = PCA(n_components=13).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T
#
#
# # account for the training data in the dataframe
# # all_properties = all_properties.drop(all_properties.tail(200).index, inplace=True)
# all_properties = all_properties.iloc[:-200]
#
#
# print(all_properties)
# print()



# balanced dataset

# # get the indices of the different types of galaxies (according to sersic index)
# spirals_indices = list(all_properties.loc[all_properties["n_r"] <= 2.5].index)
# unknown_indices = list(all_properties.loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index)
# ellipticals_indices = list(all_properties.loc[all_properties["n_r"] >= 4].index)
#
# # randomly sample half the spiral galaxies
# random.seed(1)
# chosen_spiral_indices = random.sample(spirals_indices, round(len(spirals_indices)/2))
#
# # indices of the galaxies trained on the model that we have properties for
# chosen_indices = chosen_spiral_indices + unknown_indices + ellipticals_indices
#
# # reorder the properties dataframe to match the extracted features of the balanced dataset
# all_properties = all_properties.loc[chosen_indices]
#
# # get the indices of the randomly sampled testing set (from the full dataset with augmented images)
# random.seed(2)
# dataset_size = len(chosen_spiral_indices) + len(unknown_indices) + (4 * len(ellipticals_indices))
# test_indices = random.sample(range(0, dataset_size), 20)
#
# # flag the training set in the properties dataframe (removing individually effects the position of the other elements)
# for i in test_indices:
#     if i <= len(all_properties):
#         all_properties.iloc[i] = np.nan
#
# # remove the training set from the properties dataframe
# all_properties = all_properties.dropna()
#
#
# # load the extracted features
# # extracted_features = np.load("Variational Eagle/Extracted Features/Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_300_epoch_features_" + str(run) + ".npy")[0]
# extracted_features_switch = extracted_features.T
#
# # perform pca on the extracted features
# pca = PCA(n_components=encoding_dim).fit(extracted_features)
# extracted_features = pca.transform(extracted_features)
# extracted_features_switch = extracted_features.T
#
# # get the indices of the different types of galaxies (according to sersic index) after restructuring of properties dataframe
# spirals_indices = list(all_properties.loc[all_properties["n_r"] <= 2.5].index)
# unknown_indices = list(all_properties.loc[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index)
# ellipticals_indices = list(all_properties.loc[all_properties["n_r"] >= 4].index)
#
# # split the extracted features array into the half with spirals and unknown and ellipticals
# extracted_features_spiral_unknown = extracted_features[:(len(spirals_indices) + len(unknown_indices))]
# extracted_features_elliptical = extracted_features[(len(spirals_indices) + len(unknown_indices)):]
#
# # remove the augmented images (3 of every 4 elliptical galaxies)
# extracted_features_elliptical = np.array([extracted_features_elliptical[i] for i in range(len(extracted_features_elliptical)) if i % 4 == 0])
#
# # join the two arrays back together
# extracted_features = np.array(list(extracted_features_spiral_unknown) + list(extracted_features_elliptical))
# extracted_features_switch = extracted_features.T






# print(all_properties.sort_values(by="re_r"))





# fully balanced dataset

# account for the testing dataset
all_properties = all_properties.iloc[:-200]

# load the extracted features
# extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_features_" + str(run) + ".npy")[0]
extracted_features = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_features_" + str(run) + ".npy")[0]
encoding_dim = extracted_features.shape[1]
extracted_features_switch = extracted_features.T

print(extracted_features.shape)

extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T

# perform pca on the extracted features
pca = PCA(n_components=5).fit(extracted_features)
extracted_features = pca.transform(extracted_features)
# extracted_features = extracted_features[:len(all_properties)]
extracted_features_switch = extracted_features.T






print(all_properties)






# correlation plot

# dataframe to contain correlations between each feature and each property
# correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
correlation_df = pd.DataFrame(columns=list(all_properties.columns)[1:])


# loop through each extracted feature
for feature in range(0, len(extracted_features_switch)):

    # create a list to contain the correlation between that feature and each property
    correlation_list = []

    # loop through each property
    for gal_property in range(1, len(all_properties.columns)):

        # skip the flag property
        # if gal_property == 6:
        #     continue

        # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
        correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
        correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
        correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties.iloc[:, gal_property])[0][1]
        correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]


        # add the strongest type of correlation
        correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

    # add all the correlations for that feature to the dataframe
    correlation_df.loc[len(correlation_df)] = correlation_list



print(correlation_df)



# set the figure size
# plt.figure(figsize=(20, extracted_features_switch.shape[0]))
plt.figure(figsize=(30, extracted_features_switch.shape[0]))


# properties to plot
# selected_properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]
selected_properties = ["n_r", "pa_r", "q_r", "re_r", "mag_r", "MassType_Star", "InitialMassWeightedStellarAge", "StarFormationRate", "gini", "concentration", "asymmetry", "smoothness"]

# plot a heatmap for the dataframe (with annotations)
ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})



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

    label_names = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Stellar Age", "Star Formation Rate", "Gini Coefficient", "Concentration", "Asymmetry", "Smoothness"]

    for text in label_names:
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(ax, 10)



# plt.savefig("Variational Eagle/Correlation Plots/fully_balanced_" + str(encoding_dim) + "_feature_vae_all_property_correlation_" + str(run), bbox_inches='tight')
plt.savefig("Variational Eagle/Correlation Plots/Correlation Balanced/" + str(encoding_dim) + "_feature_" + str(epochs) + "_epoch_" + str(batch_size) + "_bs_correlation_" + str(run), bbox_inches='tight')
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
# fig, axs = plt.subplots(extracted_features_switch.shape[0], len(properties), figsize=(40, (encoding_dim * 4)))
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
#     for feature in range(0, extracted_features_switch.shape[0]):
#
#         axs[feature][i].scatter(x=extracted_features_switch[feature], y=all_properties[property], s=0.5)
#         # axs[feature][i].scatter(x=extracted_features_switch[feature+19], y=all_properties[property], s=0.5)
#
#         # sns.kdeplot(data=all_properties, x=extracted_features_switch[feature], y=all_properties[property], gridsize=200)
#
#         axs[feature][i].set_xlabel("Feature " + str(feature), fontsize=12)
#         # axs[feature][i].set_xlabel("Feature " + str(feature+19), fontsize=12)
#         axs[feature][i].set_ylabel(None)
#         axs[feature][i].tick_params(labelsize=12)
#
# plt.savefig("Variational Eagle/Correlation Plots/scatter_fully_balanced_" + str(encoding_dim) + "_feature_vae_all_property_correlation_" + str(run), bbox_inches='tight')
# # plt.savefig("Variational Eagle/Correlation Plots/scatter_" + str(encoding_dim) + "_feature_all_property_correlation_p2", bbox_inches='tight')
# plt.show()







# combinations of features

# linear_model = LinearRegression()
# linear_model.fit(extracted_features, list(all_properties["mag_r"]))
#
# predicted_feature = linear_model.predict(extracted_features)
#
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# # axs.set_ylim(0, 2.5e11)
#
# plt.scatter(predicted_feature, list(all_properties["mag_r"]), s=2)
#
# plt.savefig("Variational Eagle/Plots/ab_magnitude")
# plt.show()











def quadratic(a, b, c, x):
    return (a * x * x) + (b * x) + c

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c




def density_scatter(x ,y, axs, sort=True, bins=20, **kwargs):

    # find the density colour based on the histogram
    data ,x_e, y_e = np.histogram2d(x, y, bins=bins, density=True )
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:] + y_e[:-1])), data ,np.vstack([x,y]).T, method="splinef2d", bounds_error=False)

    # replace nan with 0 (where there are no points)
    z[np.where(np.isnan(z))] = 0.0

    # sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    # make and return the scatter plot with the colour corresponding to the density
    axs.scatter(x, y, c=z, **kwargs )
    return axs






# sersic

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# # axs.scatter(x=extracted_features_switch[4], y=all_properties["n_r"], s=2)
#
# density_scatter(x=extracted_features_switch[4], y=all_properties["n_r"], axs=axs, s=5)
#
# # fit = np.polyfit(x=extracted_features_switch[4], y=all_properties["n_r"], deg=2)
# # print(fit)
# #
# # # x_fit = np.linspace(np.min(extracted_features_switch[1]), np.max(extracted_features_switch[1]), 100)
# # x_fit = np.linspace(-3, 2.5, 100)
# # y_fit = [(fit[0] * x * x) + (fit[1] * x) + fit[2] for x in x_fit]
# #
# # axs.plot(x_fit, y_fit, c="black")
#
#
# # params, covariance = curve_fit(exponential, extracted_features_switch[4], all_properties["n_r"], p0=(1, -1, 1))
# #
# # print(params)
# # print(covariance)
# #
# # x_fit = np.linspace(np.min(extracted_features_switch[4]), np.max(extracted_features_switch[4]), 100)
# # y_fit = [exponential(x, *params) for x in x_fit]
# #
# # axs.plot(x_fit, y_fit, c="red")
#
# axs.set_xlabel("PCA Feature 4")
# axs.set_ylabel("Sersic Index")
#
# # plt.savefig("Variational Eagle/Plots/pca_feature_4_vs_sersic_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()




# concentration

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# axs[0].scatter(x=extracted_features_switch[0], y=all_properties["concentration"], s=2)
# axs[0].set_title("Feature 0")
#
# axs[1].scatter(x=extracted_features_switch[4], y=all_properties["concentration"], s=2)
# axs[1].set_title("Feature 4")
#
# axs[2].scatter(-1 * predicted_feature, list(all_properties["concentration"]), s=2)
# axs[2].set_title("Weighted Sum")
#
# plt.savefig("Variational Eagle/concentration_correlation_plot")
# plt.show()





# cas parameters

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# linear_model = LinearRegression()
# linear_model.fit(extracted_features, list(all_properties["concentration"]))
# predicted_feature = linear_model.predict(extracted_features)
#
# axs[0].scatter(predicted_feature, list(all_properties["concentration"]), s=2)
# axs[0].set_title("Concentration")
# axs[0].set_ylim(2, 5.25)
#
#
# linear_model = LinearRegression()
# linear_model.fit(extracted_features, list(all_properties["asymmetry"]))
# predicted_feature = linear_model.predict(extracted_features)
#
# axs[1].scatter(predicted_feature, list(all_properties["asymmetry"]), s=2)
# axs[1].set_title("Asymmetry")
# axs[1].set_ylim(0, 0.6)
#
#
# linear_model = LinearRegression()
# linear_model.fit(extracted_features, list(all_properties["smoothness"]))
# predicted_feature = linear_model.predict(extracted_features)
#
# axs[2].scatter(predicted_feature, list(all_properties["smoothness"]), s=2)
# axs[2].set_title("Smoothness")
# axs[2].set_ylim(-0.0025, 0.11)
#
#
# plt.savefig("Variational Eagle/Plots/cas_parameters")
# plt.show()










# semi-major axis

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# axs.scatter(x=extracted_features_switch[0], y=all_properties["re_r"], s=2)
#
#
# fit = np.polyfit(x=extracted_features_switch[0], y=all_properties["re_r"], deg=2)
#
# # x_fit = np.linspace(-3, np.max(extracted_features_switch[0]), 100)
# x_fit = np.linspace(np.min(extracted_features_switch[0]), 4, 100)
# y_fit = [(fit[0] * x * x) + (fit[1] * x) + (fit[2]) for x in x_fit]
#
# axs.plot(x_fit, y_fit, c="black")
#
#
# # params, covariance = curve_fit(exponential, extracted_features_switch[0], all_properties["re_r"], p0=(1, -1, 1))
# #
# # x_fit = np.linspace(np.min(extracted_features_switch[0]), np.max(extracted_features_switch[0]), 100)
# # y_fit = [exponential(x, *params) for x in x_fit]
# #
# # axs.plot(x_fit, y_fit, c="red")
#
#
# axs.set_xlabel("PCA Feature 0")
# axs.set_ylabel("Semi-Major Axis (pkpc)")
#
# axs.set_ylim(0, 30)
#
#
# plt.savefig("Variational Eagle/Plots/pca_feature_0_vs_semi-major_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()









# position angle

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# axs.scatter(x=extracted_features_switch[3], y=all_properties["pa_r"], s=2)
#
#
# fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[3], deg=3)
# # fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[3], deg=4)
# # fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[3], deg=5)
# # fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[3], deg=6)
#
# x_fit = np.linspace(np.min(all_properties["pa_r"]), np.max(all_properties["pa_r"]), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x) + (fit[1] * x * x * x) + (fit[2] * x * x) + (fit[3] * x) + (fit[4]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x * x) + (fit[1] * x * x * x * x) + (fit[2] * x * x * x) + (fit[3] * x * x) + (fit[4] * x) + (fit[5]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x * x * x) + (fit[1] * x * x * x * x * x) + (fit[2] * x * x * x * x) + (fit[3] * x * x * x) + (fit[4] * x * x) + (fit[5] * x) + (fit[6]) for x in x_fit]
#
# plt.plot(y_fit, x_fit, c="black")
#
# axs.set_xlabel("PCA Feature 3")
# axs.set_ylabel("Position Angle")
#
#
# plt.savefig("Variational Eagle/Plots/pca_feature_3_vs_position_angle_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()





# position angle

# fig, axs = plt.subplots(2, 1, figsize=(5, 10))
#
# axs[0].scatter(x=extracted_features_switch[1], y=all_properties["pa_r"], s=2)
#
# fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[1], deg=3)
# x_fit = np.linspace(np.min(all_properties["pa_r"]), np.max(all_properties["pa_r"]), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# axs[0].plot(y_fit, x_fit, c="black")
#
# axs[0].set_xlabel("PCA Feature 1", fontsize=10)
# axs[0].set_ylabel("Position Angle (°)", fontsize=10)
# axs[0].tick_params(labelsize=10)
# axs[0].set_yticks([-90, -45, 0, 45, 90])
#
#
#
# axs[1].scatter(x=extracted_features_switch[2], y=all_properties["pa_r"], s=2)
#
# fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[2], deg=3)
# x_fit = np.linspace(np.min(all_properties["pa_r"]), np.max(all_properties["pa_r"]), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# axs[1].plot(y_fit, x_fit, c="black")
#
# axs[1].set_xlabel("PCA Feature 2", fontsize=10)
# axs[1].set_ylabel("Position Angle (°)", fontsize=10)
# axs[1].tick_params(labelsize=10)
# axs[1].set_yticks([-90, -45, 0, 45, 90])
#
# plt.savefig("Variational Eagle/Plots/pca_feature_1_and_2_vs_position_angle_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()





# position angle

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# axs.scatter(x=extracted_features_switch[1], y=all_properties["pa_r"], s=2)
#
# fit = np.polyfit(x=all_properties["pa_r"], y=extracted_features_switch[1], deg=3)
# x_fit = np.linspace(np.min(all_properties["pa_r"]), np.max(all_properties["pa_r"]), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# axs.plot(y_fit, x_fit, c="black")
#
# axs.set_xlabel("PCA Feature 1", fontsize=10)
# axs.set_ylabel("Position Angle (°)", fontsize=10)
# axs.tick_params(labelsize=10)
# axs.set_yticks([-90, -45, 0, 45, 90])
#
# plt.savefig("Variational Eagle/Plots/pca_feature_1_vs_position_angle_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()







# position angle

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# axs.scatter(x=extracted_features_switch[1], y=abs(all_properties["pa_r"]), s=2)
#
# fit = np.polyfit(x=extracted_features_switch[1], y=abs(all_properties["pa_r"]), deg=1)
# # fit = np.polyfit(x=extracted_features_switch[1], y=abs(all_properties["pa_r"]), deg=2)
# # fit = np.polyfit(x=extracted_features_switch[1], y=abs(all_properties["pa_r"]), deg=3)
#
# # x_fit = np.linspace(np.min(extracted_features_switch[1]), np.max(extracted_features_switch[1]), 100)
# x_fit = np.linspace(-2.65, 2.65, 100)
#
# y_fit = [(fit[0] * x) + (fit[1]) for x in x_fit]
# # y_fit = [(fit[0] * x * x) + (fit[1] * x) + (fit[2]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
#
# axs.plot(x_fit, y_fit, c="black")
#
# axs.set_xlabel("PCA Feature 1", fontsize=10)
# axs.set_ylabel("Position Angle (°)", fontsize=10)
# axs.tick_params(labelsize=10)
# axs.set_yticks([0, 45, 90])
#
# plt.savefig("Variational Eagle/Plots/pca_feature_1_vs_position_angle_abs_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()





# position angle

# fig, axs = plt.subplots(2, 1, figsize=(5, 10))
#
#
# axs[0].scatter(x=extracted_features_switch[2], y=abs(all_properties["pa_r"]), s=2)
#
# fit = np.polyfit(x=abs(all_properties["pa_r"]), y=extracted_features_switch[2], deg=3)
# x_fit = np.linspace(np.min(abs(all_properties["pa_r"])), np.max(abs(all_properties["pa_r"])), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# axs[0].plot(y_fit, x_fit, c="black")
#
# axs[0].set_xlabel("PCA Feature 2", fontsize=10)
# axs[0].set_ylabel("Position Angle (°)", fontsize=10)
# axs[0].tick_params(labelsize=10)
# axs[0].set_yticks([0, 45, 90])
#
#
#
# axs[1].scatter(x=extracted_features_switch[3], y=abs(all_properties["pa_r"]), s=2)
#
# fit = np.polyfit(x=abs(all_properties["pa_r"]), y=extracted_features_switch[3], deg=3)
# x_fit = np.linspace(np.min(abs(all_properties["pa_r"])), np.max(abs(all_properties["pa_r"])), 100)
# y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# axs[1].plot(y_fit, x_fit, c="black")
#
# axs[1].set_xlabel("PCA Feature 3", fontsize=10)
# axs[1].set_ylabel("Position Angle (°)", fontsize=10)
# axs[1].tick_params(labelsize=10)
# axs[1].set_yticks([0, 45, 90])
# # axs[1].set_yticks([])
#
#
# plt.savefig("Variational Eagle/Plots/pca_feature_2_and_3_vs_abs_position_angle_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()




# axis ratio

# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#
# axs.scatter(x=extracted_features_switch[2], y=all_properties["q_r"], s=2)
#
#
# # fit = np.polyfit(x=abs(extracted_features_switch[2]), y=all_properties["q_r"], deg=1)
# fit = np.polyfit(x=abs(extracted_features_switch[2]), y=all_properties["q_r"], deg=2)
# # fit = np.polyfit(x=extracted_features_switch[3], y=all_properties["q_r"], deg=3)
# # fit = np.polyfit(x=extracted_features_switch[3], y=all_properties["q_r"], deg=4)
# # fit = np.polyfit(x=extracted_features_switch[3], y=all_properties["q_r"], deg=5)
# # fit = np.polyfit(x=extracted_features_switch[3], y=all_properties["q_r"], deg=6)
#
# # x_fit = np.linspace(np.min(abs(extracted_features_switch[2])), np.max(abs(extracted_features_switch[3])), 100)
# x_fit = np.linspace(0, 4, 100)
# # y_fit = [(fit[0] * x) + (fit[1]) for x in x_fit]
# y_fit = [(fit[0] * x * x) + (fit[1] * x) + (fit[2]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x) + (fit[1] * x * x) + (fit[2] * x) + (fit[3]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x) + (fit[1] * x * x * x) + (fit[2] * x * x) + (fit[3] * x) + (fit[4]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x * x) + (fit[1] * x * x * x * x) + (fit[2] * x * x * x) + (fit[3] * x * x) + (fit[4] * x) + (fit[5]) for x in x_fit]
# # y_fit = [(fit[0] * x * x * x * x * x * x) + (fit[1] * x * x * x * x * x) + (fit[2] * x * x * x * x) + (fit[3] * x * x * x) + (fit[4] * x * x) + (fit[5] * x) + (fit[6]) for x in x_fit]
#
# plt.plot(x_fit, y_fit, c="black")
# plt.plot((-1 * x_fit), y_fit, c="black")
#
# axs.set_xlabel("PCA Feature 2")
# axs.set_ylabel("Axis Ratio")
#
#
# plt.savefig("Variational Eagle/Plots/pca_feature_2_vs_axis_ratio_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.show()





# fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# # axs[0].scatter(x=all_properties["re_r"], y=all_properties["n_r"], s=2)
# sns.kdeplot(data=all_properties, x="re_r", y="n_r", levels=20, fill=True)
# axs[0].set_xlabel("Semi-Major Axis")
# axs[0].set_ylabel("Sersic Index")
#
# axs[1].scatter(x=all_properties["asymmetry"], y=all_properties["n_r"], s=2)
# sns.kdeplot(data=all_properties, x="asymmetry", y="n_r", levels=20, fill=True)
# axs[1].set_xlabel("Asymmetry")
# axs[1].set_ylabel("Sersic Index")
#
# axs[2].scatter(x=all_properties["asymmetry"], y=all_properties["re_r"], s=2)
# axs[2].set_xlabel("Asymmetry")
# axs[2].set_ylabel("Semi-Major Axis")
#
# plt.savefig("Variational Eagle/Plots/semi-major_vs_sersic_vs_asymmetry", bbox_inches='tight')
# plt.show()



# # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# fig, axs = plt.subplots(4, 3, figsize=(15, 25))
#
# sns.kdeplot(ax = axs[0][0], data=all_properties, x="re_r", y="n_r", levels=30, gridsize=300, fill=True)
# axs[0][0].set_xlim(-2, 30)
# axs[0][0].set_ylim(-0.25, 8.25)
# # axs[0][0].set_title("All Galaxies")
# axs[0][0].set_xlabel("Semi-Major Axis")
# axs[0][0].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[1][0], data=all_properties[all_properties["n_r"] <= 2.5], x="re_r", y="n_r", levels=30, gridsize=300, fill=True)
# axs[1][0].set_xlim(-2, 30)
# axs[1][0].set_ylim(-0.25, 8.25)
# # axs[1][0].set_title("Spirals")
# axs[1][0].set_xlabel("Semi-Major Axis")
# axs[1][0].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[2][0], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="re_r", y="n_r", levels=30, gridsize=300, fill=True)
# axs[2][0].set_xlim(-2, 30)
# axs[2][0].set_ylim(-0.25, 8.25)
# # axs[2][0].set_title("Ellipticals")
# axs[2][0].set_xlabel("Semi-Major Axis")
# axs[2][0].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[3][0], data=all_properties[all_properties["n_r"] >=4], x="re_r", y="n_r", levels=30, gridsize=300, fill=True)
# axs[3][0].set_xlim(-2, 30)
# axs[3][0].set_ylim(-0.25, 8.25)
# # axs[2][0].set_title("Ellipticals")
# axs[3][0].set_xlabel("Semi-Major Axis")
# axs[3][0].set_ylabel("Sersic Index")
#
#
#
#
# sns.kdeplot(ax = axs[0][1], data=all_properties, x="asymmetry", y="n_r", levels=30, gridsize=300, fill=True)
# axs[0][1].set_xlim(-0.025, 0.5)
# axs[0][1].set_ylim(-0.25, 8.25)
# axs[0][1].set_xlabel("Asymmetry")
# axs[0][1].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[1][1], data=all_properties[all_properties["n_r"] <= 2.5], x="asymmetry", y="n_r", levels=30, gridsize=300, fill=True)
# axs[1][1].set_xlim(-0.025, 0.5)
# axs[1][1].set_ylim(-0.25, 8.25)
# axs[1][1].set_xlabel("Asymmetry")
# axs[1][1].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[2][1], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="asymmetry", y="n_r", levels=30, gridsize=300, fill=True)
# axs[2][1].set_xlim(-0.025, 0.5)
# axs[2][1].set_ylim(-0.25, 8.25)
# axs[2][1].set_xlabel("Asymmetry")
# axs[2][1].set_ylabel("Sersic Index")
#
# sns.kdeplot(ax = axs[3][1], data=all_properties[all_properties["n_r"] >=4], x="asymmetry", y="n_r", levels=30, gridsize=300, fill=True)
# axs[3][1].set_xlim(-0.025, 0.5)
# axs[3][1].set_ylim(-0.25, 8.25)
# axs[3][1].set_xlabel("Asymmetry")
# axs[3][1].set_ylabel("Sersic Index")
#
#
#
#
# sns.kdeplot(ax = axs[0][2], data=all_properties, x="re_r", y="asymmetry", levels=30, gridsize=300, fill=True)
# axs[0][2].set_xlim(-2, 30)
# axs[0][2].set_ylim(-0.025, 0.5)
# axs[0][2].set_xlabel("Semi-Major Axis")
# axs[0][2].set_ylabel("Asymmetry")
#
# sns.kdeplot(ax = axs[1][2], data=all_properties[all_properties["n_r"] <= 2.5], x="re_r", y="asymmetry", levels=30, gridsize=300, fill=True)
# axs[1][2].set_xlim(-2, 30)
# axs[1][2].set_ylim(-0.025, 0.5)
# axs[1][2].set_xlabel("Semi-Major Axis")
# axs[1][2].set_ylabel("Asymmetry")
#
# sns.kdeplot(ax = axs[2][2], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="re_r", y="asymmetry", levels=30, gridsize=300, fill=True)
# axs[2][2].set_xlim(-2, 30)
# axs[2][2].set_ylim(-0.025, 0.5)
# axs[2][2].set_xlabel("Semi-Major Axis")
# axs[2][2].set_ylabel("Asymmetry")
#
# sns.kdeplot(ax = axs[3][2], data=all_properties[all_properties["n_r"] >=4], x="re_r", y="asymmetry", levels=30, gridsize=300, fill=True)
# axs[3][2].set_xlim(-2, 30)
# axs[3][2].set_ylim(-0.025, 0.5)
# axs[3][2].set_xlabel("Semi-Major Axis")
# axs[3][2].set_ylabel("Asymmetry")
#
#
# plt.savefig("Variational Eagle/Plots/semi-major_vs_sersic_vs_asymmetry_density_3", bbox_inches='tight')
# plt.show()








# # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# fig, axs = plt.subplots(4, 3, figsize=(15, 25))
#
# sns.scatterplot(ax = axs[0][0], data=all_properties, x="re_r", y="n_r", edgecolor=None, s=2, legend=False)
# axs[0][0].set_xlim(-2, 30)
# axs[0][0].set_ylim(-0.25, 8.25)
# # axs[0][0].set_title("All Galaxies")
# axs[0][0].set_xlabel("Semi-Major Axis")
# axs[0][0].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[1][0], data=all_properties[all_properties["n_r"] <= 2.5], x="re_r", y="n_r", edgecolor=None, s=2, legend=False)
# axs[1][0].set_xlim(-2, 30)
# axs[1][0].set_ylim(-0.25, 8.25)
# # axs[1][0].set_title("Spirals")
# axs[1][0].set_xlabel("Semi-Major Axis")
# axs[1][0].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[2][0], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="re_r", y="n_r", edgecolor=None, s=5, legend=False)
# axs[2][0].set_xlim(-2, 30)
# axs[2][0].set_ylim(-0.25, 8.25)
# # axs[2][0].set_title("Ellipticals")
# axs[2][0].set_xlabel("Semi-Major Axis")
# axs[2][0].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[3][0], data=all_properties[all_properties["n_r"] >=4], x="re_r", y="n_r", edgecolor=None, s=10, legend=False)
# axs[3][0].set_xlim(-2, 30)
# axs[3][0].set_ylim(-0.25, 8.25)
# # axs[2][0].set_title("Ellipticals")
# axs[3][0].set_xlabel("Semi-Major Axis")
# axs[3][0].set_ylabel("Sersic Index")
#
#
#
#
# sns.scatterplot(ax = axs[0][1], data=all_properties, x="asymmetry", y="n_r", edgecolor=None, s=2, legend=False)
# axs[0][1].set_xlim(-0.025, 0.5)
# axs[0][1].set_ylim(-0.25, 8.25)
# axs[0][1].set_xlabel("Asymmetry")
# axs[0][1].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[1][1], data=all_properties[all_properties["n_r"] <= 2.5], x="asymmetry", y="n_r", edgecolor=None, s=2, legend=False)
# axs[1][1].set_xlim(-0.025, 0.5)
# axs[1][1].set_ylim(-0.25, 8.25)
# axs[1][1].set_xlabel("Asymmetry")
# axs[1][1].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[2][1], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="asymmetry", y="n_r", edgecolor=None, s=5, legend=False)
# axs[2][1].set_xlim(-0.025, 0.5)
# axs[2][1].set_ylim(-0.25, 8.25)
# axs[2][1].set_xlabel("Asymmetry")
# axs[2][1].set_ylabel("Sersic Index")
#
# sns.scatterplot(ax = axs[3][1], data=all_properties[all_properties["n_r"] >= 4], x="asymmetry", y="n_r", edgecolor=None, s=10, legend=False)
# axs[3][1].set_xlim(-0.025, 0.5)
# axs[3][1].set_ylim(-0.25, 8.25)
# axs[3][1].set_xlabel("Asymmetry")
# axs[3][1].set_ylabel("Sersic Index")
#
#
#
#
# sns.scatterplot(ax = axs[0][2], data=all_properties, x="re_r", y="asymmetry", edgecolor=None, s=2, legend=False)
# axs[0][2].set_xlim(-2, 30)
# axs[0][2].set_ylim(-0.025, 0.5)
# axs[0][2].set_xlabel("Semi-Major Axis")
# axs[0][2].set_ylabel("Asymmetry")
#
# sns.scatterplot(ax = axs[1][2], data=all_properties[all_properties["n_r"] <= 2.5], x="re_r", y="asymmetry", edgecolor=None, s=2, legend=False)
# axs[1][2].set_xlim(-2, 30)
# axs[1][2].set_ylim(-0.025, 0.5)
# axs[1][2].set_xlabel("Semi-Major Axis")
# axs[1][2].set_ylabel("Asymmetry")
#
# sns.scatterplot(ax = axs[2][2], data=all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")], x="re_r", y="asymmetry", edgecolor=None, s=5, legend=False)
# axs[2][2].set_xlim(-2, 30)
# axs[2][2].set_ylim(-0.025, 0.5)
# axs[2][2].set_xlabel("Semi-Major Axis")
# axs[2][2].set_ylabel("Asymmetry")
#
# sns.scatterplot(ax = axs[3][2], data=all_properties[all_properties["n_r"] >=4], x="re_r", y="asymmetry", edgecolor=None, s=10, legend=False)
# axs[3][2].set_xlim(-2, 30)
# axs[3][2].set_ylim(-0.025, 0.5)
# axs[3][2].set_xlabel("Semi-Major Axis")
# axs[3][2].set_ylabel("Asymmetry")
#
#
# plt.savefig("Variational Eagle/Plots/semi-major_vs_sersic_vs_asymmetry_scatter", bbox_inches='tight')
# plt.show()








# all_properties.loc[list(all_properties[all_properties["n_r"] <= 2.5].index), "sersic_cut"] = "Spiral"
# all_properties.loc[list(all_properties[all_properties["n_r"] >= 4].index), "sersic_cut"] = "Elliptical"
# all_properties.loc[list(all_properties[all_properties["n_r"].between(2.5, 4, inclusive="neither")].index), "sersic_cut"] = "Unknown"
#
# print(all_properties)
#
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#
# sns.histplot(ax = axs[0], data=all_properties, x="n_r", hue="sersic_cut", stat="density", common_norm=False, element="poly", bins=20, fill=True)
# # sns.histplot(ax = axs[0], data=all_properties, x="n_r", hue="sersic_cut", element="poly", bins=10)
# sns.histplot(ax = axs[1], data=all_properties, x="re_r", hue="sersic_cut", stat="density", common_norm=False, element="poly", bins=15, fill=True)
# sns.histplot(ax = axs[2], data=all_properties, x="asymmetry", hue="sersic_cut", stat="density", common_norm=False, element="poly", bins=20, fill=True)
#
# # sns.histplot(ax = axs[0], data=all_properties, x="n_r", hue="sersic_cut", stat="density", common_norm=False, bins=15)
# # sns.histplot(ax = axs[1], data=all_properties, x="re_r", hue="sersic_cut", stat="density", common_norm=False, bins=15)
# # sns.histplot(ax = axs[2], data=all_properties, x="asymmetry", hue="sersic_cut", stat="density", common_norm=False, bins=15)
#
# axs[0].set_xlabel("Sersic Index")
# axs[1].set_xlabel("Semi-Major Axis")
# axs[2].set_xlabel("Asymmetry")
#
#
# plt.savefig("Variational Eagle/Plots/sersic_semi-major_asymmetry_histogram_3", bbox_inches='tight')
# plt.show()

