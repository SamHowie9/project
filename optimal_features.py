import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap



# encoding_dim = 24
#
# extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")



df1 = pd.read_csv("stab3510_supplemental_file/table1.csv", comment="#")


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")

# print(df1["GalaxyID"].tolist())
# print(df1)

# print(structure_properties)
# print(physical_properties)
# print(all_properties)




# structual features: Sersic index, Axis ratio, position angle, semi-major axis

# physical propeties: stellar mass, star formation rate, halo mass, black hole mass, merger history

# stellar mass, gas mass, dm_mass, bh_mass (subgrid), mean age, sfr,
# MassType_Star, Mass_Type_Gas, Mass_Type_DM, Mass_Type_BH, BlackHoleMass, InitialMassWeightedStellarAge, StarFormationRate



loss = []
val_loss = []

med_loss = []
max_loss = []
min_loss = []

med_val_loss = []
max_val_loss = []
min_val_loss = []

# 16, 26, 36    25, 41
# 25, 35, 45    1, 8, 27, 35


for i in range(1, 46):

    feature_loss = np.load("Loss/" + str(i) + "_feature_loss_3.npy")

    # loss.append(feature_loss[0])
    # val_loss.append(feature_loss[1])

    feature_loss_1 = np.load("Loss/" + str(i) + "_feature_loss.npy")
    feature_loss_2 = np.load("Loss/" + str(i) + "_feature_loss_2.npy")
    feature_loss_3 = np.load("Loss/" + str(i) + "_feature_loss_3.npy")

    med_loss.append(np.median((feature_loss_1[0], feature_loss_2[0], feature_loss_3[0])))
    max_loss.append(max(feature_loss_1[0], feature_loss_2[0], feature_loss_3[0]))
    min_loss.append(min(feature_loss_1[0], feature_loss_2[0], feature_loss_3[0]))

    med_val_loss.append(np.median((feature_loss_1[1], feature_loss_2[1], feature_loss_3[1])))
    max_val_loss.append(max(feature_loss_1[1], feature_loss_2[1], feature_loss_3[1]))
    min_val_loss.append(min(feature_loss_1[1], feature_loss_2[1], feature_loss_3[1]))


loss_err = []
val_loss_err = []

for i in range(len(med_loss)):

    loss_err.append([(med_loss[i] - min_loss[i]), (max_loss[i] - med_loss[i])])
    val_loss_err.append([(med_val_loss[i] - min_val_loss[i]), (max_val_loss[i] - med_val_loss[i])])

loss_err = np.array(loss_err).T
val_loss_err = np.array(val_loss_err).T

# print(loss_err)
# print(val_loss_err)

plt.figure(figsize=(10, 8))

plt.scatter(x=range(1, 46), y=med_loss, label="Training Images")
plt.errorbar(x=range(1, 46), y=med_loss, yerr=loss_err, ls="none", capsize=3)

plt.scatter(x=range(1, 46), y=med_val_loss, label="Validation Images")
plt.errorbar(x=range(1, 46), y=med_val_loss, yerr=val_loss_err, ls="none", capsize=3)

plt.xlabel("Extracted Features", fontsize=15)
plt.ylabel("Loss", fontsize=15)

plt.savefig("Plots/extracted_feat_vs_loss")
plt.show()


# plt.scatter(x=range(1, 46), y=loss, label="Training Images")
# plt.scatter(x=range(1, 46), y=val_loss, label="Validation Images")
#
# # plt.plot(range(17, 41), loss)
# # plt.plot(range(17, 41), val_loss)
#
# plt.xlabel("Number of Extracted Features")
# plt.ylabel("Root-Mean-Squared Error")
#
# plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2)
#
# # plt.savefig("Plots/extracted_feat_vs_loss")
# plt.show()








relevant_feature_number = []
relevant_feature_ratio = []

med_relevant_feature_number = []
max_relevant_feature_number = []
min_relevant_feature_number = []

med_relevant_feature_ratio = []
max_relevant_feature_ratio = []
min_relevant_feature_ratio = []


for encoding_dim in range(1, 46):

    extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")
    extracted_features_switch = np.flipud(np.rot90(extracted_features))

    structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])

    extracted_features_1 = np.load("Features/" + str(encoding_dim) + "_features.npy")
    extracted_features_2 = np.load("Features/" + str(encoding_dim) + "_features_2.npy")
    extracted_features_3 = np.load("Features/" + str(encoding_dim) + "_features_3.npy")

    extracted_features_switch_1 = np.flipud(np.rot90(extracted_features_1))
    extracted_features_switch_2 = np.flipud(np.rot90(extracted_features_2))
    extracted_features_switch_3 = np.flipud(np.rot90(extracted_features_3))

    correlation_df_1 = pd.DataFrame(columns=all_properties.columns[1:])
    correlation_df_2 = pd.DataFrame(columns=all_properties.columns[1:])
    correlation_df_3 = pd.DataFrame(columns=all_properties.columns[1:])

    # print(correlation_df_1)

    for feature in range(0, len(extracted_features_switch)):

        # create a list to contain the correlation between that feature and each property
        correlation_list = []

        correlation_list_1 = []
        correlation_list_2 = []
        correlation_list_3 = []

        # loop through each property
        for gal_property in range(1, len(all_properties.columns)):

            # calculate the correlation between that extracted feature and that property
            # correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
            correlation = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_list.append(correlation)

            correlation_1 = np.corrcoef(extracted_features_switch_1[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_2 = np.corrcoef(extracted_features_switch_2[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_3 = np.corrcoef(extracted_features_switch_3[feature], all_properties.iloc[:, gal_property])[0][1]

            correlation_list_1.append(correlation_1)
            correlation_list_2.append(correlation_2)
            correlation_list_3.append(correlation_3)



        # add the correlation of that feature to the main dataframe
        # structure_correlation_df.loc[len(structure_correlation_df)] = correlation_list

        # print(correlation_list_1)

        correlation_df_1.loc[len(correlation_df_1)] = correlation_list_1
        correlation_df_2.loc[len(correlation_df_2)] = correlation_list_2
        correlation_df_3.loc[len(correlation_df_3)] = correlation_list_3

    # find the number of features at least slightly correlating with a property
    relevant_features = (abs(structure_correlation_df).max(axis=1) > 0.2).sum()

    relevant_features_1 = (abs(correlation_df_1).max(axis=1) > 0.2).sum()
    relevant_features_2 = (abs(correlation_df_2).max(axis=1) > 0.2).sum()
    relevant_features_3 = (abs(correlation_df_3).max(axis=1) > 0.2).sum()



    relevant_feature_number.append(relevant_features)
    relevant_feature_ratio.append(relevant_features/encoding_dim)

    med_relevant_feature_number.append(np.median((relevant_features_1, relevant_features_2, relevant_features_3)))
    max_relevant_feature_number.append(max(relevant_features_1, relevant_features_2, relevant_features_3))
    min_relevant_feature_number.append(min(relevant_features_1, relevant_features_2, relevant_features_3))

    med_relevant_feature_ratio.append(np.median((relevant_features_1/encoding_dim, relevant_features_2/encoding_dim, relevant_features_3/encoding_dim)))
    max_relevant_feature_ratio.append(max((relevant_features_1/encoding_dim), (relevant_features_2/encoding_dim), (relevant_features_3/encoding_dim)))
    min_relevant_feature_ratio.append(min((relevant_features_1/encoding_dim), (relevant_features_2/encoding_dim), (relevant_features_3/encoding_dim)))



# print(med_relevant_feature_number)
# print(max_relevant_feature_number)
# print(min_relevant_feature_number)

relevant_err = []
ratio_err = []

for i in range(len(med_relevant_feature_number)):
    relevant_err.append([(med_relevant_feature_number[i] - min_relevant_feature_number[i]), (max_relevant_feature_number[i] - med_relevant_feature_number[i])])
    ratio_err.append([(med_relevant_feature_ratio[i] - min_relevant_feature_ratio[i]), (max_relevant_feature_ratio[i] - med_relevant_feature_ratio[i])])

relevant_err = np.array(relevant_err).T
ratio_err = np.array(ratio_err).T



plt.figure(figsize=(10, 8))

plt.scatter(x=range(1, 46), y=med_relevant_feature_number)
plt.errorbar(x=range(1, 46), y=med_relevant_feature_number, yerr=relevant_err, ls="none", capsize=3)

# plt.scatter(x=range(1, 46), y=relevant_feature_number)



plt.xlabel("Total Number of Extracted Features", fontsize=15)
plt.ylabel("Number of Meaningful Extracted Features", fontsize=15)

plt.savefig("Plots/meaningful_extracted_features")
plt.show()


plt.figure(figsize=(10, 8))

plt.scatter(x=range(1, 46), y=med_relevant_feature_ratio)
plt.errorbar(x=range(1, 46), y=med_relevant_feature_ratio, yerr=ratio_err, ls="none", capsize=3)

plt.xlabel("Total Number of Extracted Features", fontsize=15)
plt.ylabel("Ratio of Meaningful to Total Number", fontsize=15)

plt.savefig("Plots/meaningful_extracted_features_ratio")
plt.show()





# sns.histplot(data=structure_properties, x="q_r", bins=300)
# plt.show()








# encoding_dim=44
#
#
# extracted_features = np.load("Features/" + str(encoding_dim) + "_features.npy")
#
# extracted_features_switch = np.flipud(np.rot90(extracted_features))
#
# structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])
# physical_correlation_df = pd.DataFrame(columns=["Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
# correlation_df = pd.DataFrame(columns=["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
#
# for feature in range(0, len(extracted_features_switch)):
#
#     # create a list to contain the correlation between that feature and each property
#     structure_correlation_list = []
#     physical_correlation_list = []
#
#     # loop through each property
#     for gal_property in range(1, len(structure_properties.columns)):
#
#         # calculate the correlation between that extracted feature and that property
#         structure_correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
#         structure_correlation_list.append(structure_correlation)
#
#     for gal_property in range(1, len(physical_properties.columns)):
#
#         # calculate the correlation between that extracted feature and that property
#         physical_correlation = np.corrcoef(extracted_features_switch[feature], physical_properties.iloc[:, gal_property])[0][1]
#         physical_correlation_list.append(physical_correlation)
#
#     # add the correlation of that feature to the main dataframe
#     structure_correlation_df.loc[len(structure_correlation_df)] = structure_correlation_list
#     physical_correlation_df.loc[len(physical_correlation_df)] = physical_correlation_list
#     correlation_df.loc[len(correlation_df)] = structure_correlation_list + physical_correlation_list
#
#
# print(structure_correlation_df)
# print(physical_correlation_df)
# print(correlation_df)







# # set the figure size
# plt.figure(figsize=(12, 16))
#
# # sns.set(font_scale = 2)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(structure_correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=15)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
# ax.figure.axes[-1].yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots/" + str(encoding_dim) + "_feature_structure_property_correlation")
# plt.show()
#
#
#
#
# # set the figure size
# plt.figure(figsize=(15, 16))
#
# # sns.set(font_scale = 2)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(physical_correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=15)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
# ax.figure.axes[-1].yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots/" + str(encoding_dim) + "_feature_physical_property_correlation")
# plt.show()






# # set the figure size
# plt.figure(figsize=(20, 16))
#
# # sns.set(font_scale = 2)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=15)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
# ax.figure.axes[-1].yaxis.label.set_size(15)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=15)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots/" + str(encoding_dim) + "_feature_all_property_correlation")
# plt.show()
