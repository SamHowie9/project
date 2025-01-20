import numpy as np
import pandas as pd
import scipy.optimize
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import textwrap
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA




pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



plt.style.use("default")
sns.set_style("ticks")




# # load supplemental file (find galaxies with stellar mass > 10^10 solar masses
# df1 = pd.read_csv("Galaxy Properties/Eagle Properties/stab3510_supplemental_file/table1.csv", comment="#")
#
# # load structural and physical properties into dataframes
# structure_properties = pd.read_csv("Galaxy Properties/Eagle Properties/structure_propeties.csv", comment="#")
# physical_properties = pd.read_csv("Galaxy Properties/Eagle Properties/physical_properties.csv", comment="#")
#
# # account for hte validation data and remove final 200 elements
# structure_properties.drop(structure_properties.tail(200).index, inplace=True)
# physical_properties.drop(physical_properties.tail(200).index, inplace=True)
#
# # dataframe for all properties
# all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")











# dataframe containing all losses
# df_loss = pd.DataFrame(columns=["Extracted Features", "Min Loss", "Min KL", "Med Loss", "Med KL", "Max Loss", "Max KL"])
df_loss = pd.DataFrame(columns=["Extracted Features", "Min Total", "Med Total", "Max Total", "Min Reconstruction", "Med Reconstruction", "Max Reconstruction", "Min KL", "Med KL", "Max KL"])

for i in range(5, 31):
    try:

        # load the three different runs
        loss_1 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_300_epoch_loss_1.npy"))
        loss_2 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_300_epoch_loss_2.npy"))
        loss_3 = list(np.load("Variational Eagle/Loss/Fully Balanced/" + str(i) + "_feature_300_epoch_loss_3.npy"))

        # sort the reconstruction loss and kl divergence
        total_sorted = np.sort(np.array([loss_1[0], loss_2[0], loss_3[0]]))
        reconstruction_sorted = np.sort(np.array([loss_1[1], loss_2[1], loss_3[1]]))
        kl_sorted = np.sort(np.array([loss_1[2], loss_2[2], loss_3[2]]))

        # dataframe to store order of losses (reconstruction and kl divergence)
        # df_loss.loc[len(df_loss)] = [i, loss_sorted[0], kl_sorted[0], loss_sorted[1], kl_sorted[1], loss_sorted[2], kl_sorted[2]]
        df_loss.loc[len(df_loss)] = [i] + list(total_sorted) + list(reconstruction_sorted) + list(kl_sorted)

    # if we don't have a run for this number of features, skip it
    except:
        print(i)

print(df_loss)


# find the size of the loss error bars for total loss
total_err_upper = np.array(df_loss["Max Total"] - df_loss["Med Total"])
total_err_lower = np.array(df_loss["Med Total"] - df_loss["Min Total"])

# find the size of the loss error bars for reconstruction loss
reconstruction_err_upper = np.array(df_loss["Max Reconstruction"] - df_loss["Med Reconstruction"])
reconstruction_err_lower = np.array(df_loss["Med Reconstruction"] - df_loss["Min Reconstruction"])

# find the size of the loss error bars for kl divergence
kl_err_upper = np.array(df_loss["Max KL"] - df_loss["Med KL"])
kl_err_lower = np.array(df_loss["Med KL"] - df_loss["Min KL"])





# def logfit(x, a, b):
#     return a * np.log10(x) + b
#
# params, covarience = curve_fit(logfit, range(1, 51), df_loss["Med Loss"])
# a, b = params

# def sigmoid_fit(x, a, b, c, d):
#     return (a/(1 + (np.e ** (-b*x + c)))) + d
#
# params, covarience = curve_fit(sigmoid_fit, range(1, 51), df_loss["Med Loss"]/1000)
# a, b, c, d = params

#
# x_fit = np.linspace(1, 50, 100).tolist()
# y_fit = []
#
# for x in x_fit:
#     y_fit.append(sigmoid_fit(x, a, b, c, d)*1000)
#
# axs[0].plot(x_fit, y_fit, c="black")




fig, axs = plt.subplots(1, 1, figsize=(8, 4))
# fig, axs = plt.subplots(2, 1, figsize=(12, 10))

axs.errorbar(df_loss["Extracted Features"], df_loss["Med Total"], yerr=[total_err_lower, total_err_upper], fmt="o")
axs.set_ylabel("Loss")
axs.set_xlabel("Extracted Features")

plt.show()


# loss = axs[0].errorbar(df_loss["Extracted Features"], df_loss["Med Loss"], yerr=[loss_err_lower, loss_err_upper], fmt="o", label="Loss")
# axs[0].set_ylabel("Loss")
# axs[0].set_xlabel("Extracted Features")
#
# # axs2 = axs[0].twinx()
# axs2 = axs[1]
#
# # axs2.plot(x_fit, y_fit, c="black")
#
# # axs2.plot(df_loss["Extracted Features"], df_loss["Med KL"], color="red")
# kl_div = axs2.errorbar(df_loss["Extracted Features"], df_loss["Med KL"], yerr=[kl_err_lower, kl_err_upper], fmt="o", color="red", label="KL-Divergence")
# axs2.set_ylabel("KL-Divergence")
#
# axs[0].legend([loss, kl_div], ["Loss", "KL-Divergence"], loc="center right")
#
#
# plt.savefig("Variational Eagle/Plots/loss_plot_fully_balanced")
# plt.show()





# fig, axs = plt.subplots(1, 1, figsize=(10, 5))
#
# df_num = pd.DataFrame(columns=["Extracted Features", "Min", "Med", "Max"])
#
# for i in range(1, 31):
#
#     features_1 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_300_epoch_features_1.npy")[0]
#     features_2 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_300_epoch_features_2.npy")[0]
#     features_3 = np.load("Variational Eagle/Extracted Features/Fully Balanced/" + str(i) + "_feature_300_epoch_features_3.npy")[0]
#
#     pca_1 = PCA(n_components=0.999).fit(features_1)
#     pca_2 = PCA(n_components=0.999).fit(features_2)
#     pca_3 = PCA(n_components=0.999).fit(features_3)
#
#     num_1 = pca_1.components_.shape[0]
#     num_2 = pca_2.components_.shape[0]
#     num_3 = pca_3.components_.shape[0]
#
#     sorted = np.sort(np.array([num_1, num_2, num_3]))
#
#     df_num.loc[len(df_num)] = [i, sorted[0], sorted[1], sorted[2]]
#
# # find the size of the loss error bars for reconstruction loss
# num_err_upper = np.array(df_num["Max"] - df_num["Med"])
# num_err_lower = np.array(df_num["Med"] - df_num["Min"])
#
# axs.errorbar(df_num["Extracted Features"], df_num["Med"], yerr=[num_err_lower, num_err_upper], fmt="o")
#
# plt.show()






# relevant_feature_number = []
# relevant_feature_ratio = []
#
# med_relevant_feature_number = []
# max_relevant_feature_number = []
# min_relevant_feature_number = []
#
# relevant_property_count = []
#
# # med_relevant_feature_ratio = []
# # max_relevant_feature_ratio = []
# # min_relevant_feature_ratio = []
#
# for encoding_dim in range(1, 31):
#
#     extracted_features = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]
#     extracted_features_switch = np.flipud(np.rot90(extracted_features))
#
#     structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])
#
#
#     extracted_features_1 = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_1.npy")[0]
#     extracted_features_2 = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_2.npy")[0]
#     extracted_features_3 = np.load("Variational Eagle/Extracted Features/Normalised Individually/" + str(encoding_dim) + "_feature_300_epoch_features_3.npy")[0]
#     # extracted_features_2 = np.load("Variational Eagle/Extracted Features/" + str(encoding_dim) + "_feature_300_epoch_features_2.npy")[0]
#     # extracted_features_3 = np.load("Variational Eagle/Extracted Features/" + str(encoding_dim) + "_feature_300_epoch_features_3.npy")[0]
#
#     extracted_features_switch_1 = np.flipud(np.rot90(extracted_features_1))
#     extracted_features_switch_2 = np.flipud(np.rot90(extracted_features_2))
#     extracted_features_switch_3 = np.flipud(np.rot90(extracted_features_3))
#
#     correlation_df_1 = pd.DataFrame(columns=all_properties.columns[1:])
#     correlation_df_2 = pd.DataFrame(columns=all_properties.columns[1:])
#     correlation_df_3 = pd.DataFrame(columns=all_properties.columns[1:])
#
#     # print(correlation_df_1)
#
#     for feature in range(0, len(extracted_features_switch)):
#
#         # create a list to contain the correlation between that feature and each property
#         correlation_list = []
#
#         correlation_list_1 = []
#         correlation_list_2 = []
#         correlation_list_3 = []
#
#         # loop through each property
#         for gal_property in range(1, len(all_properties.columns)):
#
#             # calculate the correlation between that extracted feature and that property
#             # correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
#             # correlation = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
#             # correlation_list.append(correlation)
#
#             correlation_1_1 = np.corrcoef(extracted_features_switch_1[feature], all_properties.iloc[:, gal_property])[0][1]
#             correlation_1_2 = np.corrcoef(extracted_features_switch_1[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_1_3 = np.corrcoef(abs(extracted_features_switch_1[feature]), all_properties.iloc[:, gal_property])[0][1]
#             correlation_1_4 = np.corrcoef(abs(extracted_features_switch_1[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_1 = max(abs(correlation_1_1), abs(correlation_1_2), abs(correlation_1_3), abs(correlation_1_4))
#
#             correlation_2_1 = np.corrcoef(extracted_features_switch_2[feature], all_properties.iloc[:, gal_property])[0][1]
#             correlation_2_2 = np.corrcoef(extracted_features_switch_2[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_2_3 = np.corrcoef(abs(extracted_features_switch_2[feature]), all_properties.iloc[:, gal_property])[0][1]
#             correlation_2_4 = np.corrcoef(abs(extracted_features_switch_2[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_2 = max(abs(correlation_2_1), abs(correlation_2_2), abs(correlation_2_3), abs(correlation_2_4))
#
#             correlation_3_1 = np.corrcoef(extracted_features_switch_3[feature], all_properties.iloc[:, gal_property])[0][1]
#             correlation_3_2 = np.corrcoef(extracted_features_switch_3[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_3_3 = np.corrcoef(abs(extracted_features_switch_3[feature]), all_properties.iloc[:, gal_property])[0][1]
#             correlation_3_4 = np.corrcoef(abs(extracted_features_switch_3[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
#             correlation_3 = max(abs(correlation_3_1), abs(correlation_3_2), abs(correlation_3_3), abs(correlation_3_4))
#
#
#             # correlation_1 = np.corrcoef(extracted_features_switch_1[feature], all_properties.iloc[:, gal_property])[0][1]
#             # correlation_2 = np.corrcoef(extracted_features_switch_2[feature], all_properties.iloc[:, gal_property])[0][1]
#             # correlation_3 = np.corrcoef(extracted_features_switch_3[feature], all_properties.iloc[:, gal_property])[0][1]
#
#
#             correlation_list_1.append(correlation_1)
#             correlation_list_2.append(correlation_2)
#             correlation_list_3.append(correlation_3)
#
#         correlation_df_1.loc[len(correlation_df_1)] = correlation_list_1
#         correlation_df_2.loc[len(correlation_df_2)] = correlation_list_2
#         correlation_df_3.loc[len(correlation_df_3)] = correlation_list_3
#
#
#     relevant_properties = ["n_r", "q_r", "re_r", "mag_r", "MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]
#     # relevant_properties = ["StarFormationRate"]
#
#     # find the number of features at least slightly correlating with a property
#     relevant_features = (abs(structure_correlation_df).max(axis=1) > 0.4).sum()
#
#     relevant_features_1 = (abs(correlation_df_1[relevant_properties]).max(axis=1) > 0.4).sum()
#     relevant_features_2 = (abs(correlation_df_2[relevant_properties]).max(axis=1) > 0.4).sum()
#     relevant_features_3 = (abs(correlation_df_3[relevant_properties]).max(axis=1) > 0.4).sum()
#
#     relevant_feature_number.append(relevant_features)
#     # relevant_feature_ratio.append(relevant_features/encoding_dim)
#
#     med_relevant_feature_number.append(np.median((relevant_features_1, relevant_features_2, relevant_features_3)))
#     max_relevant_feature_number.append(max(relevant_features_1, relevant_features_2, relevant_features_3))
#     min_relevant_feature_number.append(min(relevant_features_1, relevant_features_2, relevant_features_3))
#
#     # med_relevant_feature_ratio.append(np.median((relevant_features_1/encoding_dim, relevant_features_2/encoding_dim, relevant_features_3/encoding_dim)))
#     # max_relevant_feature_ratio.append(max((relevant_features_1/encoding_dim), (relevant_features_2/encoding_dim), (relevant_features_3/encoding_dim)))
#     # min_relevant_feature_ratio.append(min((relevant_features_1/encoding_dim), (relevant_features_2/encoding_dim), (relevant_features_3/encoding_dim)))
#
#
#     # for property in relevant_properties:
#     #
#     #     relevant_features_1 = (abs(correlation_df_1[property]).max(axis=1) > 0.4).sum()
#     #     relevant_features_2 = (abs(correlation_df_2[property]).max(axis=1) > 0.4).sum()
#     #     relevant_features_3 = (abs(correlation_df_3[property]).max(axis=1) > 0.4).sum()
#     #
#     #     med_relevant_feature_number.append(np.median((relevant_features_1, relevant_features_2, relevant_features_3)))
#     #     max_relevant_feature_number.append(max(relevant_features_1, relevant_features_2, relevant_features_3))
#     #     min_relevant_feature_number.append(min(relevant_features_1, relevant_features_2, relevant_features_3))
#     #
#     #     relevant_property_count.append([min_relevant_feature_number, med_relevant_feature_number, max_relevant_feature_number])
#
#
#
# axs[1].errorbar(range(1, 31), med_relevant_feature_number, yerr=[np.array(med_relevant_feature_number) - np.array(min_relevant_feature_number), np.array(max_relevant_feature_number) - np.array(med_relevant_feature_number)], fmt="o")
# # axs[1].errorbar(range(1, 51), relevant_property_count[0][1], yerr=[relevant_property_count[0][1] - relevant_property_count[0][1]), np.array(max_relevant_feature_number) - np.array(med_relevant_feature_number)], fmt="o")
# axs[1].set_ylabel("Meaningful Extracted Features")
# axs[1].set_xlabel("Extracted Features")
#
# # plt.savefig("Variational Eagle/Plots/Meaningful Extracted Features vs Total Extracted Features")
# # plt.show()
#
#
#
# plt.savefig("Variational Eagle/Plots/Individual Normalisation - Optimal Extracted Features")
# plt.show()






# relevant_err = []
# ratio_err = []
#
# for i in range(len(med_relevant_feature_number)):
#     relevant_err.append([(med_relevant_feature_number[i] - min_relevant_feature_number[i]), (max_relevant_feature_number[i] - med_relevant_feature_number[i])])
#     ratio_err.append([(med_relevant_feature_ratio[i] - min_relevant_feature_ratio[i]), (max_relevant_feature_ratio[i] - med_relevant_feature_ratio[i])])
#
# relevant_err = np.array(relevant_err).T
# ratio_err = np.array(ratio_err).T
#
#
#
# # plt.figure(figsize=(10, 8))
#
# # x_values = range(1, 46)
#
# # plt.scatter(x=x_values, y=med_relevant_feature_number)
# # plt.errorbar(x=x_values, y=med_relevant_feature_number, yerr=relevant_err, ls="none", capsize=3, alpha=0.6)
#
# # sns.lmplot(x=list(range(1, 46))*3, y=(min_relevant_feature_number + med_relevant_feature_number + max_relevant_feature_number))
#
# df = pd.DataFrame()
# df["Extracted Features"] = list(range(1, 51))*3
# df["med_relevant_feature_number"] = min_relevant_feature_number + med_relevant_feature_number + max_relevant_feature_number
# print(df)

# sns.set_style("ticks")

# plt.figure(figsize=(10, 8))


# with sns.axes_style("ticks"):
#     sns.lmplot(data=df, x="Extracted Features", y="med_relevant_feature_number", logx=True, ci=0, height=8, aspect=1.5, line_kws={"color": "black"}, scatter_kws={"s": 0})
    # sns.lmplot(ax=axs[1], data=df, x="Extracted Features", y="med_relevant_feature_number", logx=True, ci=0, height=8, aspect=1.5, line_kws={"color": "black"}, scatter_kws={"s": 0})

# with sns.axes_style("ticks"):
#     sns.lmplot(data=df, x="Extracted Features", y="med_relevant_feature_number", order=2, ci=0, height=8, aspect=1.25, line_kws={"color": "red"}, scatter_kws={"s": 0})

# logfit = scipy.optimize.curve_fit(lambda t, a, b: a*np.log(t), range(1, 51), med_relevant_feature_number, p0=(0.6, 2.3))
# a = logfit[0]
# b = logfit[1]
#
# yfit = []
#
# for i in range(1, 51):
#     yfit.append(a * np.log(i))
#
# print(logfit)


# plt.plot(range(1, 51), yfit, c="red")


# sns.despine(left=False, bottom=False, top=False, right=False)



# plt.scatter(x=range(1, 51), y=med_relevant_feature_number)
# plt.errorbar(x=range(1, 51), y=med_relevant_feature_number, yerr=relevant_err, ls="none", alpha=0.6)
#
# plt.xlabel("Total Extracted Features", fontsize=20)
# plt.ylabel("Meaningful Extracted Features", fontsize=20)
#
# plt.tick_params(labelsize=20)



# def logfit(x, a, b):
#     return a * np.log10(x) + b
#
# params, covarience = curve_fit(logfit, range(1, 51), med_relevant_feature_number)
#
# a, b = params
#
# print("Logfit params: " + str(a) + " " + str(b))
#
# x_fit = np.linspace(1, 50, 100).tolist()
#
# print(x_fit)
#
# y_fit = []
#
# for x in x_fit:
#     y_fit.append(logfit(x, a, b))
#
#
#
#
# plt.figure(figsize=(12, 5))
#
# plt.plot(x_fit, y_fit, c="black")
#
# plt.scatter(x=range(1, 51), y=med_relevant_feature_number, c=colours_blue)
# plt.errorbar(x=range(1, 51), y=med_relevant_feature_number, yerr=relevant_err, ecolor=colours_blue, ls="none", alpha=0.6)
#
# plt.xlabel("Total Extracted Features", fontsize=18)
# plt.ylabel("Meaningful Extracted Features", fontsize=18)
#
# plt.tick_params(labelsize=18)
#
# plt.savefig("Plots/rand_meaningful_extracted_features_0-3_abs", bbox_inches='tight')
# plt.show()




# axs[1].plot(x_fit, y_fit, c="black")
#
# axs[1].scatter(x=range(1, 51), y=med_relevant_feature_number, c=colours_blue)
# axs[1].errorbar(x=range(1, 51), y=med_relevant_feature_number, yerr=relevant_err, ecolor=colours_blue, ls="none", alpha=0.6)
#
# axs[1].set_xlabel("Total Extracted Features", fontsize=18)
# axs[1].set_ylabel("Meaningful Extracted Features", fontsize=18)
#
# axs[1].tick_params(labelsize=15)
#
# fig.tight_layout()
#
# plt.savefig("Plots/optimal_extracted_features")
# plt.show()









# plt.figure(figsize=(10, 8))
#
# plt.scatter(x=range(1, 46), y=med_relevant_feature_ratio)
# plt.errorbar(x=range(1, 46), y=med_relevant_feature_ratio, yerr=ratio_err, ls="none", capsize=3, alpha=0.6)
#
# plt.xlabel("Total Number of Extracted Features", fontsize=15)
# plt.ylabel("Ratio of Meaningful to Total Extracted Features", fontsize=15)
#
# plt.tick_params(labelsize=12)
#
# plt.savefig("Plots/rand_meaningful_extracted_features_ratio_abs")
# plt.show()





# sns.histplot(data=structure_properties, x="q_r", bins=300)
# plt.show()
