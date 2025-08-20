import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
import random
import dcor
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
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


# 16, 25

run = 18
encoding_dim = 35
n_flows = 0
beta = 0.0001
beta_name = "0001"
epochs = 750
batch_size = 32






def density_scatter(x ,y, axs, **kwargs):

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axs.scatter(x, y, c=z, **kwargs)
    return axs







# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_real.csv")
# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_balanced.csv")

# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_spirals.csv")
all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_ellipticals.csv")
# all_properties = pd.read_csv("Galaxy Properties/Eagle Properties/all_properties_transitional.csv")


print(all_properties.shape)



# print(all_properties_real[all_properties_real["DiscToTotal"] < 0].shape)
# print(all_properties_real[all_properties_real["DiscToTotal"] > 1].shape)

# print(all_properties[all_properties["GalaxyID"] == 8407169])
# print(all_properties[all_properties["GalaxyID"] == 8756517])
# print(all_properties[all_properties["GalaxyID"] == 8937440])
# print(all_properties[all_properties["GalaxyID"] == 8827412])
#
# print()
#
# print(all_properties[all_properties["GalaxyID"] == 16618997])
# print(all_properties[all_properties["GalaxyID"] == 17171464])
# print(all_properties[all_properties["GalaxyID"] == 13632283])
# print(all_properties[all_properties["GalaxyID"] == 18481115])
#
#
# print()
#
# print(all_properties[all_properties["GalaxyID"] == 8274107])
# print(all_properties[all_properties["GalaxyID"] == 8101596])
# print(all_properties[all_properties["GalaxyID"] == 15583095])










# for n_flows in [n_flows]:
# for run in [16, 25]:
# for run in range(1, 11):
# for run in range(1, 26):
# for encoding_dim in [encoding_dim]:
# for run in [2, 5, 7, 10, 12, 15, 17, 18, 19, 20, 22, 23]:
for run in [4, 9, 11, 12, 14, 15, 18, 19, 23, 25]:

    print(n_flows, run)

    # load the extracted features
    extracted_features = np.load("Variational Eagle/Extracted Features/Ellipticals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")
    # extracted_features = np.load("Variational Eagle/Extracted Features/Spirals/planar_new_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_default_transformed.npy")

    encoding_dim = extracted_features.shape[1]

    print(extracted_features.shape)




    # perform pca on the extracted features
    pca = PCA(n_components=0.999, svd_solver="full").fit(extracted_features)
    extracted_features = pca.transform(extracted_features)

    print(extracted_features.shape)





    # spirals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] > 0.2].index.tolist()
    # # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] > 0.2]



    # ellipticals only
    # spiral_indices = all_properties[all_properties["DiscToTotal"] < 0.1].index.tolist()
    # # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"] < 0.1]


    # transitional only
    # spiral_indices = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")].index.tolist()
    # # print(spiral_indices)
    # extracted_features = extracted_features[spiral_indices]
    # all_properties = all_properties[all_properties["DiscToTotal"].between(0.1, 0.2, inclusive="both")]



    # original images
    # all_properties = all_properties_real
    # extracted_features = extracted_features[:len(all_properties)]





    # correlation plot

    # dataframe to contain correlations between each feature and each property
    correlation_df = pd.DataFrame(columns=list(all_properties.columns)[1:])

    # max_corr = []

    # loop through each extracted feature
    for feature in range(0, extracted_features.shape[1]):

        # create a list to contain the correlation between that feature and each property
        correlation_list = []

        # loop through each property
        for gal_property in range(1, all_properties.shape[1]):

            # calculate the correlation coefficients (multiple for different types of correlation eg. mirrored)
            # correlation_1 = np.corrcoef(extracted_features.T[feature], all_properties.iloc[:, gal_property])[0][1]
            # correlation_2 = np.corrcoef(extracted_features.T[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
            # correlation_3 = np.corrcoef(abs(extracted_features.T[feature]), all_properties.iloc[:, gal_property])[0][1]
            # correlation_4 = np.corrcoef(abs(extracted_features.T[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
            #
            # # add the strongest type of correlation
            # correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

            correlation = dcor.distance_correlation(extracted_features.T[feature], all_properties.iloc[:, gal_property])
            correlation_list.append(correlation)

        print(len(correlation_list))

        # # keep correlations of relevant properties
        # selected_property_indices = [64, 1, 3, 2, 31, 38, 39, 40, 7, 9, 10, 11, 12, 13]
        #
        # # add the maximum correlation to the list
        # max_corr.append(max([correlation_list[index-1] for index in selected_property_indices]))

        # add all the correlations for that feature to the dataframe
        correlation_df.loc[len(correlation_df)] = correlation_list

    # set index so feature label starts at 1 rather than 0
    correlation_df.index = correlation_df.index + 1

    # np.save("Variational Eagle/Correlation Plots/Normalising Flows Balanced/Normal/max_corr.npy", max_corr)


    # set the figure size
    plt.figure(figsize=(22, correlation_df.shape[0]))


    # properties to plot
    # selected_properties = ["n_r", "DiscToTotal", "re_r", "rhalf_ellip", "pa_r", "q_r",  "mag_r", "MassType_Star", "InitialMassWeightedStellarAge", "StarFormationRate", "gini", "m20", "concentration", "asymmetry", "smoothness"]
    # selected_properties = ["DiscToTotal", "pa_r", "rhalf_ellip", "n_r", "q_r", "concentration", "asymmetry", "smoothness"]
    selected_properties = ["DiscToTotal", "n_r", "q_r", "pa_r", "rhalf_ellip", "concentration", "asymmetry", "smoothness", "g-r", "g-i"]
    # selected_properties = ["MassType_Star", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]


    # plot a heatmap for the dataframe (with annotations)
    # ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
    ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, annot_kws={"size":15}, cmap="Blues", vmin=0, vmax=0.8, cbar_kws={"label": "Correlation", "pad": 0.02, "aspect": 60})



    plt.yticks(rotation=0)
    plt.ylabel("Extracted Features", fontsize=15)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0, labelsize=15)
    ax.figure.axes[-1].yaxis.label.set_size(15)

    colourbar = ax.collections[0].colorbar
    colourbar.ax.tick_params(labelsize=15)
    colourbar.ax.yaxis.label.set_size(15)


    def wrap_labels(ax, width, break_long_words=False):

        labels = []
        # for label in ax.get_xticklabels():
            # text = label.get_text()

        # label_names = ["Sersic Index", "Disk-Total Ratio", "Semi - Major Axis", "Half Light Radius", "Position Angle", "Axis Ratio", "AB Magnitude", "Stellar Mass", "Stellar Age", "Star Formation Rate", "Gini Coefficient", "M20", "Concentration", "Asymmetry", "Smoothness"]
        # label_names = ["D/T", "Position Angle", "Half Light Radius", "Sersic Index", "Axis Ratio", "Concentration", "Asymmetry", "Smoothness"]
        label_names = ["D/T", "Sérsic Index", "Axis Ratio", "Position Angle", "Half-Light Radius", "Concentration", "Asymmetry", "Smoothness", "g-r", "g-i"]
        # label_names = ["Stellar Mass", "Dark Matter Mass", "Black Hole Mass (Particle)", "Black Hole Mass (Subgrid)", "Mean Stellar Age", "Star Formation Rate"]

        # selected_properties = ["MassType_Star", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]

        for text in label_names:
            labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0, fontsize=15)

    wrap_labels(ax, 10)


    plt.savefig("Variational Eagle/Correlation Plots/Ellipticals/PCA/structure_measurement_correlation_" + str(run), bbox_inches='tight')
    # plt.savefig("Variational Eagle/Correlation Plots/Spirals/PCA/_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced_dcor", bbox_inches='tight')
    plt.show(block=False)
    plt.close()







    # all density plots

    # # scale font on plots
    # # default_size = plt.rcParams['font.size']
    # # plt.rcParams.update({'font.size': default_size * 2})
    #
    # all_properties["pa_r"] = abs(all_properties["pa_r"])
    #
    # selected_properties = ["DiscToTotal", "n_r", "q_r", "pa_r", "rhalf_ellip", "concentration", "asymmetry", "smoothness"]
    # property_names = ["D/T", "Sérsic Index", "Axis Ratio", "Position Angle", "Half-Light Radius", "Concentration", "Asymmetry", "Smoothness"]
    #
    # # selected_properties = ["MassType_Star", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]
    # # property_names = ["Stellar Mass", "Dark Matter Mass", "Black Hole Mass (Particle)", "Black Hole Mass (Subgrid)", "Mean Stellar Age", "Star Formation Rate"]
    #
    # fig, axs = plt.subplots(len(selected_properties), extracted_features.shape[1], figsize=(extracted_features.shape[1]*5, len(selected_properties)*5))
    #
    # for i, property in enumerate(selected_properties):
    #
    #     axs[i][0].set_ylabel(property_names[i])
    #
    #     for j in range(extracted_features.shape[1]):
    #
    #         if i == 0:
    #
    #             axs[0][j].set_title("Feature " + str(j+1))
    #
    #         # density_scatter(x=extracted_features.T[i], y=all_properties[property], axs=axs[i][j])
    #         axs[i][j].scatter(x=extracted_features.T[j], y=all_properties[property], s=1)
    #
    # fig.subplots_adjust(wspace=0.2, hspace=0.2)
    #
    # plt.savefig("Variational Eagle/Correlation Plots/Normalising Flows Balanced/Normal/scatter_latent_" + str(encoding_dim) + "_beta_" + beta_name + "_epoch_" + str(epochs) + "_flows_" + str(n_flows) + "_" + str(run) + "_balanced", bbox_inches="tight")
    # plt.show()











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





















#
# fig, axs = plt.subplots(1, 1, figsize=(12, 10))
#
# density_scatter(extracted_features.T[4], all_properties["rhalf_ellip"], axs=axs, s=5)
# axs.set_xlabel("Feature 5")
# axs.set_ylabel("Half-Light Radius")
#
# plt.show()



# fig, axs = plt.subplots(1, 1, figsize=(12, 10))
#
# density_scatter(all_properties["n_r"], all_properties["DiscToTotal"], axs=axs, s=5)
# axs.set_xlabel("Sersic Index")
# axs.set_ylabel("Disk-Total Ratio")
#
# plt.show()




# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
#
# density_scatter(extracted_features.T[0], all_properties["n_r"], axs=axs[0], s=5)
# axs[0].set_xlabel("Feature 1")
# axs[0].set_ylabel("Disk-Total Ratio")
#
# density_scatter(extracted_features.T[0], all_properties["DiscToTotal"], axs=axs[1], s=5)
# axs[1].set_xlabel("Feature 3")
# axs[1].set_ylabel("Disk-Total Ratio")
#
# density_scatter(extracted_features.T[9], all_properties["n_r"], axs=axs[2], s=5)
# axs[2].set_xlabel("Feature 6")
# axs[2].set_ylabel("Disk-Total Ratio")
#
#
# plt.show()



# fig, axs = plt.subplots(1, 1, figsize=(8, 7))
#
# sns.histplot(x=all_properties["DiscToTotal"], ax=axs, bins=50)
# axs.set_xlabel("Disk-Total Ratio")
#
# plt.savefig("Variational Eagle/Plots/disk_total_distribution")
# plt.show()






# structure measurement comparison

# fig, axs = plt.subplots(1, 5, figsize=(25, 5))
#
# density_scatter(extracted_features.T[0], all_properties["n_r"], axs=axs[0], s=2)
# axs[0].set_title("Sersic Index")
# axs[0].set_xlabel("Feature 1")
# axs[0].set_ylabel("Sersic Index")
#
# density_scatter(extracted_features.T[2], all_properties["pa_r"], axs=axs[1], s=2)
# axs[1].set_title("Gini Coefficient")
# axs[1].set_xlabel("Feature 3")
# axs[1].set_ylabel("Gini Coefficient")
#
# density_scatter(extracted_features.T[2], all_properties["q_r"], axs=axs[2], s=2)
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
# plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_" + str(encoding_dim) + "_" + str(run), bbox_inches='tight')
# plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_" + str(encoding_dim) + "_" + str(run) + ".pdf", bbox_inches='tight')
# plt.show()







    # fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    #
    # density_scatter(extracted_features.T[0], all_properties["concentration"], axs=axs[0][0], s=2)
    # # axs[0][0].set_title("Concentration")
    # axs[0][0].set_xlabel("Feature 1")
    # axs[0][0].set_ylabel("Concentration")
    # # axs[0][0].set_xlim(-4, 4)
    # # axs[0][0].set_ylim(0, 6)
    #
    # density_scatter(extracted_features.T[1], abs(all_properties["concentration"]), axs=axs[0][1], s=2)
    # # axs[0][1].set_title("Concentration")
    # axs[0][1].set_xlabel("Feature 2")
    # axs[0][1].set_ylabel("Concentration")
    # # axs[0][1].set_yticks([0, 45, 90])
    # # axs[0][1].set_xlim(-3.5, 3.5)
    #
    # density_scatter(extracted_features.T[9], all_properties["concentration"], axs=axs[0][2], s=2)
    # # axs[0][2].set_title("Concentration")
    # axs[0][2].set_xlabel("Feature 10")
    # axs[0][2].set_ylabel("Concentration")
    # # axs[0][2].set_xlim(-3, 3)
    #
    #
    #
    #
    # density_scatter(extracted_features.T[0], all_properties["asymmetry"], axs=axs[1][0], s=2)
    # # axs[1][0].set_title("Asymmetry")
    # axs[1][0].set_xlabel("Feature 1")
    # axs[1][0].set_ylabel("Asymmetry")
    # # axs[1][1].set_xlim(-3, 3)
    # # axs[1][1].set_ylim(2, 5)
    #
    # density_scatter(extracted_features.T[1], abs(all_properties["asymmetry"]), axs=axs[1][1], s=2)
    # # axs[1][1].set_title("Asymmetry")
    # axs[1][1].set_xlabel("Feature 2")
    # axs[1][1].set_ylabel("Asymmetry")
    # # axs[1][2].set_xlim(-4, 4)
    # # axs[1][2].set_ylim(0, 0.5)
    #
    # density_scatter(extracted_features.T[9], abs(all_properties["smoothness"]), axs=axs[1][2], s=2)
    # # axs[1][2].set_title("Asymmetry")
    # axs[1][2].set_xlabel("Feature 10")
    # axs[1][2].set_ylabel("Asymmetry")
    # # axs[1][3].set_xlim(-4, 4)
    # # axs[1][3].set_ylim(0, 0.1)
    #
    #
    #
    #
    # density_scatter(extracted_features.T[0], all_properties["smoothness"], axs=axs[2][0], s=2)
    # # axs[1][0].set_title("Smoothness")
    # axs[1][0].set_xlabel("Feature 1")
    # axs[1][0].set_ylabel("Smoothness")
    # # axs[1][1].set_xlim(-3, 3)
    # # axs[1][1].set_ylim(2, 5)
    #
    # density_scatter(extracted_features.T[1], abs(all_properties["smoothness"]), axs=axs[2][1], s=2)
    # # axs[1][1].set_title("Smoothness")
    # axs[1][1].set_xlabel("Feature 2")
    # axs[1][1].set_ylabel("Smoothness")
    # # axs[1][2].set_xlim(-4, 4)
    # # axs[1][2].set_ylim(0, 0.5)
    #
    # density_scatter(extracted_features.T[9], abs(all_properties["smoothness"]), axs=axs[2][2], s=2)
    # # axs[1][2].set_title("Smoothness")
    # axs[1][2].set_xlabel("Feature 10")
    # axs[1][2].set_ylabel("Smoothness")
    # # axs[1][3].set_xlim(-4, 4)
    # # axs[1][3].set_ylim(0, 0.1)
    #
    #
    #
    # plt.show()






    # # scale font on plots
    # default_size = plt.rcParams['font.size']
    # plt.rcParams.update({'font.size': default_size * 2})
    #
    # fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    #
    # density_scatter(extracted_features.T[1], all_properties["n_r"], axs=axs[0][0], s=3)
    # # axs[0][0].set_title("Sersic Index")
    # axs[0][0].set_xlabel("Feature 2")
    # axs[0][0].set_ylabel("Sérsic Index")
    # axs[0][0].set_xlim(-3.5, 4)
    # axs[0][0].set_ylim(0, 6)
    #
    # density_scatter(extracted_features.T[3], abs(all_properties["pa_r"]), axs=axs[0][1], s=3)
    # # axs[0][1].set_title("Position Angle")
    # axs[0][1].set_xlabel("Feature 4")
    # axs[0][1].set_ylabel("Position Angle (°)")
    # axs[0][1].set_yticks([0, 45, 90])
    # axs[0][1].set_xlim(-3.5, 3.5)
    #
    # density_scatter(extracted_features.T[5], all_properties["q_r"], axs=axs[0][2], s=3)
    # # axs[0][2].set_title("Axis Ratio")
    # axs[0][2].set_xlabel("Feature 6")
    # axs[0][2].set_ylabel("Axis Ratio")
    # axs[0][2].set_xlim(-3, 3)
    #
    # # density_scatter(extracted_features.T[3], all_properties["m20"], axs=axs[0][3], s=10)
    # # axs[0][3].set_title("M20")
    # # axs[0][3].set_xlabel("Feature 0")
    # # axs[0][3].set_ylabel("M20")
    # # # axs[0][3].set_xlim(-6, 4)
    # # # axs[0][3].set_ylim(-2.5, -1.2)
    # #
    # # density_scatter(extracted_features.T[4], all_properties["gini"], axs=axs[1][0], s=10)
    # # axs[1][0].set_title("Gini Coefficient")
    # # axs[1][0].set_xlabel("Feature 4")
    # # axs[1][0].set_ylabel("Gini Coefficient")
    # # # axs[1][0].set_xlim(-4, 4)
    # # # axs[1][0].set_ylim(0.4, 0.65)
    #
    # density_scatter(extracted_features.T[1], all_properties["concentration"], axs=axs[1][0], s=3)
    # # axs[1][0].set_title("Concentration")
    # axs[1][0].set_xlabel("Feature 2")
    # axs[1][0].set_ylabel("Concentration")
    # axs[1][0].set_xlim(-3.5, 3.5)
    # axs[1][0].set_ylim(2, 5)
    #
    # density_scatter(extracted_features.T[0], abs(all_properties["asymmetry"]), axs=axs[1][1], s=3)
    # # axs[1][1].set_title("Asymmetry")
    # axs[1][1].set_xlabel("Feature 1")
    # axs[1][1].set_ylabel("Asymmetry")
    # axs[1][1].set_xlim(-4, 4)
    # axs[1][1].set_ylim(0, 0.5)
    #
    # density_scatter(extracted_features.T[0], abs(all_properties["smoothness"]), axs=axs[1][2], s=3)
    # # axs[1][2].set_title("Smoothness")
    # axs[1][2].set_xlabel("Feature 1")
    # axs[1][2].set_ylabel("Smoothness")
    # axs[1][2].set_xlim(-4, 4)
    # axs[1][2].set_ylim(0, 0.1)
    #
    #
    # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    #
    #
    # plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_" + str(encoding_dim) + "_" + str(run) + "_zoomed", bbox_inches='tight')
    # plt.savefig("Variational Eagle/Plots/structure_measurement_comparisons_" + str(encoding_dim) + "_" + str(run) + "_zoomed.pdf", bbox_inches='tight')
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





