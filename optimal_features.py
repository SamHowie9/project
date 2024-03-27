import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import textwrap
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# encoding_dim = 25
#
# extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_1.npy")


plt.style.use("default")
sns.set_style("ticks")


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

# 1, 34, 44
# 9, 19, 20, 26, 34, 35, 45


for i in range(1, 46):

    # feature_loss = np.load("Loss Rand/" + str(i) + "_feature_loss_1.npy")
    #
    # loss.append(feature_loss[0])
    # val_loss.append(feature_loss[1])

    feature_loss_1 = np.load("Loss Rand/" + str(i) + "_feature_loss_1.npy")
    feature_loss_2 = np.load("Loss Rand/" + str(i) + "_feature_loss_2.npy")
    feature_loss_3 = np.load("Loss Rand/" + str(i) + "_feature_loss_3.npy")

    if i == 23 or i == 26 or i == 45:
        print(feature_loss_1[0])
        print(feature_loss_2[0])
        print(feature_loss_3[0])
        print()

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

print(loss_err)
print(val_loss_err)

plt.figure(figsize=(12, 8))




plt.scatter(x=range(1, 46), y=np.exp(med_loss), label="Training Images", zorder=10)
# plt.errorbar(x=range(1, 46), y=med_loss, yerr=loss_err, ls="none", capsize=3, alpha=0.6, zorder=0)

# plt.scatter(x=range(1, 46), y=med_val_loss, label="Validation Images", zorder=11)
# plt.errorbar(x=range(1, 46), y=med_val_loss, yerr=val_loss_err, ls="none", capsize=3, alpha=0.6, zorder=1)

plt.xlabel("Extracted Features", fontsize=20)
plt.ylabel("Loss", fontsize=20)

plt.tick_params(labelsize=20)

# plt.grid(False)



plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc='lower center', ncol=2, prop={"size":20})

plt.savefig("Plots/rand_extracted_feat_vs_loss", bbox_inches='tight')
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

    extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_1.npy")
    extracted_features_switch = np.flipud(np.rot90(extracted_features))

    structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])

    extracted_features_1 = np.load("Features Rand/" + str(encoding_dim) + "_features_1.npy")
    extracted_features_2 = np.load("Features Rand/" + str(encoding_dim) + "_features_2.npy")
    extracted_features_3 = np.load("Features Rand/" + str(encoding_dim) + "_features_3.npy")

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
            # correlation = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
            # correlation_list.append(correlation)

            correlation_1_1 = np.corrcoef(extracted_features_switch_1[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_1_2 = np.corrcoef(extracted_features_switch_1[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_1_3 = np.corrcoef(abs(extracted_features_switch_1[feature]), all_properties.iloc[:, gal_property])[0][1]
            correlation_1_4 = np.corrcoef(abs(extracted_features_switch_1[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_1 = max(abs(correlation_1_1), abs(correlation_1_2), abs(correlation_1_3), abs(correlation_1_4))

            correlation_2_1 = np.corrcoef(extracted_features_switch_2[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_2_2 = np.corrcoef(extracted_features_switch_2[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_2_3 = np.corrcoef(abs(extracted_features_switch_2[feature]), all_properties.iloc[:, gal_property])[0][1]
            correlation_2_4 = np.corrcoef(abs(extracted_features_switch_2[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_2 = max(abs(correlation_2_1), abs(correlation_2_2), abs(correlation_2_3), abs(correlation_2_4))

            correlation_3_1 = np.corrcoef(extracted_features_switch_3[feature], all_properties.iloc[:, gal_property])[0][1]
            correlation_3_2 = np.corrcoef(extracted_features_switch_3[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_3_3 = np.corrcoef(abs(extracted_features_switch_3[feature]), all_properties.iloc[:, gal_property])[0][1]
            correlation_3_4 = np.corrcoef(abs(extracted_features_switch_3[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]
            correlation_3 = max(abs(correlation_3_1), abs(correlation_3_2), abs(correlation_3_3), abs(correlation_3_4))


            # correlation_1 = np.corrcoef(extracted_features_switch_1[feature], all_properties.iloc[:, gal_property])[0][1]
            # correlation_2 = np.corrcoef(extracted_features_switch_2[feature], all_properties.iloc[:, gal_property])[0][1]
            # correlation_3 = np.corrcoef(extracted_features_switch_3[feature], all_properties.iloc[:, gal_property])[0][1]

            correlation_list_1.append(correlation_1)
            correlation_list_2.append(correlation_2)
            correlation_list_3.append(correlation_3)



        # add the correlation of that feature to the main dataframe
        # structure_correlation_df.loc[len(structure_correlation_df)] = correlation_list

        # print(correlation_list_1)

        correlation_df_1.loc[len(correlation_df_1)] = correlation_list_1
        correlation_df_2.loc[len(correlation_df_2)] = correlation_list_2
        correlation_df_3.loc[len(correlation_df_3)] = correlation_list_3



    relevant_properties = ["n_r", "q_r", "re_r", "mag_r", "MassType_Star", "MassType_Gas", "MassType_DM", "MassType_BH", "BlackHoleMass", "InitialMassWeightedStellarAge", "StarFormationRate"]


    # find the number of features at least slightly correlating with a property
    relevant_features = (abs(structure_correlation_df).max(axis=1) > 0.3).sum()

    relevant_features_1 = (abs(correlation_df_1[relevant_properties]).max(axis=1) > 0.3).sum()
    relevant_features_2 = (abs(correlation_df_2[relevant_properties]).max(axis=1) > 0.3).sum()
    relevant_features_3 = (abs(correlation_df_3[relevant_properties]).max(axis=1) > 0.3).sum()

    # if encoding_dim == 40:
    #     print(relevant_features_1)
    #     print(relevant_features_2)
    #     print(relevant_features_3)
    #     print()

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



# plt.figure(figsize=(10, 8))

# x_values = range(1, 46)

# plt.scatter(x=x_values, y=med_relevant_feature_number)
# plt.errorbar(x=x_values, y=med_relevant_feature_number, yerr=relevant_err, ls="none", capsize=3, alpha=0.6)

# sns.lmplot(x=list(range(1, 46))*3, y=(min_relevant_feature_number + med_relevant_feature_number + max_relevant_feature_number))

df = pd.DataFrame()
df["Extracted Features"] = list(range(1, 46))*3
df["med_relevant_feature_number"] = min_relevant_feature_number + med_relevant_feature_number + max_relevant_feature_number
print(df)

# sns.set_style("ticks")

# plt.figure(figsize=(10, 8))

with sns.axes_style("ticks"):
    sns.lmplot(data=df, x="Extracted Features", y="med_relevant_feature_number", logx=True, ci=0, height=8, aspect=1.5, line_kws={"color": "black"}, scatter_kws={"s": 0})

# with sns.axes_style("ticks"):
#     sns.lmplot(data=df, x="Extracted Features", y="med_relevant_feature_number", order=2, ci=0, height=8, aspect=1.25, line_kws={"color": "black"}, scatter_kws={"s": 0})


sns.despine(left=False, bottom=False, top=False, right=False)

plt.scatter(x=range(1, 46), y=med_relevant_feature_number)
plt.errorbar(x=range(1, 46), y=med_relevant_feature_number, yerr=relevant_err, ls="none", capsize=3, alpha=0.6)


plt.xlabel("Total Extracted Features", fontsize=20)
plt.ylabel("Meaningful Extracted Features", fontsize=20)

plt.tick_params(labelsize=20)


# sns.lmplot(data=all_properties, x=list(range(1, 46)), y=med_relevant_feature_number)


# fit = np.polyfit(x=np.log(x_values), y=med_relevant_feature_number, deg=1)
#
# y_fit = fit[0] * np.log(x_values) + fit[1]
#
# plt.plot(x_values, y_fit, c="black")



# linear = np.polyfit(x=x_values, y=np.log(med_relevant_feature_number), deg=1)
# quadratic = np.polyfit(x=x_values, y=np.log(med_relevant_feature_number), deg=2)
# cubic = np.polyfit(x=x_values, y=np.log(med_relevant_feature_number), deg=3)
#
# v1 = np.polyval(linear, x_values)
# v2 = np.polyval(quadratic, x_values)
# v3 = np.polyval(cubic, x_values)
#
# plt.plot(x_values, v1, c="black")
# plt.plot(x_values, v2, c="yellow")
# plt.plot(x_values, v3, c="red")



# plt.scatter(x=range(1, 46), y=relevant_feature_number)

plt.savefig("Plots/rand_meaningful_extracted_features_0-3_abs", bbox_inches='tight')
plt.show()



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
