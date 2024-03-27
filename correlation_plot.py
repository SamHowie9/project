import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


encoding_dim = 32


extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_3.npy")
extracted_features_switch = np.flipud(np.rot90(extracted_features))



# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")


print(all_properties)



# model = AgglomerativeClustering()
#
# visualizer = KElbowVisualizer(model, k=(1, 40), timings=False)
#
# visualizer.fit(extracted_features)        # Fit the data to the visualizer
# visualizer.show()







structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])
physical_correlation_df = pd.DataFrame(columns=["Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])

for feature in range(0, len(extracted_features_switch)):

    # create a list to contain the correlation between that feature and each property
    structure_correlation_list = []
    physical_correlation_list = []

    correlation_list = []

    # loop through each property
    # for gal_property in range(1, len(structure_properties.columns)):
    for gal_property in range(1, len(all_properties.columns)):

        # calculate the correlation between that extracted feature and that property
        # structure_correlation = np.corrcoef(extracted_features_switch[feature], abs(structure_properties.iloc[:, gal_property]))[0][1]
        # structure_correlation_list.append(structure_correlation)

        correlation_1 = np.corrcoef(extracted_features_switch[feature], all_properties.iloc[:, gal_property])[0][1]
        correlation_2 = np.corrcoef(extracted_features_switch[feature], abs(all_properties.iloc[:, gal_property]))[0][1]
        correlation_3 = np.corrcoef(abs(extracted_features_switch[feature]), all_properties.iloc[:, gal_property])[0][1]
        correlation_4 = np.corrcoef(abs(extracted_features_switch[feature]), abs(all_properties.iloc[:, gal_property]))[0][1]

        if gal_property == 1 and feature == 21:
            print(correlation_1, correlation_2, correlation_3, correlation_4)

        correlation_list.append(max(abs(correlation_1), abs(correlation_2), abs(correlation_3), abs(correlation_4)))

        # # linear, quadratic and cubic fitting
        # linear = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=1)
        # quadratic = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=2)
        # cubic = np.polyfit(x=extracted_features_switch[feature], y=all_properties.iloc[:, gal_property], deg=3)
        #
        # # evaluate these polynomials
        # val_linear = np.polyval(linear, extracted_features_switch[feature])
        # val_quadratic = np.polyval(quadratic, extracted_features_switch[feature])
        # val_cubic = np.polyval(cubic, extracted_features_switch[feature])
        #
        # # mean squared error between polynomial fits and data
        # mse_linear = np.mean(np.square(all_properties.iloc[:, gal_property] - val_linear))
        # mse_quadratic = np.mean(np.square(all_properties.iloc[:, gal_property] - val_quadratic))
        # mse_cubic = np.mean(np.square(all_properties.iloc[:, gal_property] - val_cubic))
        #
        # best_fit = min(mse_linear, mse_quadratic, mse_cubic)
        #
        # correlation_list.append(best_fit)



    # for gal_property in range(1, len(physical_properties.columns)):
    #
    #     # calculate the correlation between that extracted feature and that property
    #     physical_correlation = np.corrcoef(extracted_features_switch[feature], physical_properties.iloc[:, gal_property])[0][1]
    #     physical_correlation_list.append(physical_correlation)

    # # add the correlation of that feature to the main dataframe
    # structure_correlation_df.loc[len(structure_correlation_df)] = structure_correlation_list
    # physical_correlation_df.loc[len(physical_correlation_df)] = physical_correlation_list
    # correlation_df.loc[len(correlation_df)] = structure_correlation_list + physical_correlation_list

    # print(correlation_list)
    #
    correlation_df.loc[len(correlation_df)] = correlation_list




# print(structure_correlation_df)
# print(physical_correlation_df)
# print(correlation_df)








# # set the figure size
# plt.figure(figsize=(12, 16))
#
# sns.set(font_scale=1.5)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Sersic Index", "Position Angle", "Axis Ratio"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation Coefficient'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=22, labelpad=10)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
#
# ax.figure.axes[-1].yaxis.label.set_size(22)
# ax.figure.axes[-1].yaxis.labelpad = 15
#
#
# # colorbar = plt.colorbar(ax)
# # colorbar.setlabel("Correlation Coefficient", fontsize=22, labelpad=10)
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=22)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_structure_property_correlation_2.eps")
# plt.show()




# # set the figure size
# plt.figure(figsize=(20, 16))
#
# sns.set(font_scale=1.5)
#
# # plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation Coefficient'})
#
#
# plt.yticks(rotation=0)
# plt.ylabel("Extracted Features", fontsize=22)
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.tick_params(length=0)
#
# ax.figure.axes[-1].yaxis.label.set_size(22)
# ax.figure.axes[-1].yaxis.labelpad = 15
#
#
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0, fontsize=22)
#
# wrap_labels(ax, 10)
#
#
#
# plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_physical_property_correlation_2")
# plt.show()


selected_properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]

print(correlation_df)


# set the figure size
plt.figure(figsize=(20, 16))

# sns.set(font_scale = 2)

# ["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"]

# plot a heatmap for the dataframe (with annotations)
# ax = sns.heatmap(abs(correlation_df[["Sersic Index", "Semi - Major Axis", "AB Magnitude", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
ax = sns.heatmap(abs(correlation_df[selected_properties]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})



plt.yticks(rotation=0)
plt.ylabel("Extracted Features", fontsize=15)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(length=0)
ax.figure.axes[-1].yaxis.label.set_size(15)


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=15)

wrap_labels(ax, 10)



plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_all_property_correlation_3_abs")
plt.show()



# plt.figure(figsize=(10,10))
# plt.scatter(x=extracted_features_switch[9], y=all_properties["q_r"])
# plt.show()





# "n_r", "pa_r", "q_r", "re_r", "mag_r", "MassType_Star", "MassType_DM", "MassType_BH", "InitialMassWeightedStellarAge", "StarFormationRate"

# properties = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass", "Stellar Age", "Star Formation Rate"]
properties = ["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]


all_properties = all_properties[["n_r", "pa_r", "q_r", "re_r", "InitialMassWeightedStellarAge", "StarFormationRate", "MassType_Star", "MassType_DM", "MassType_BH"]]

print(all_properties)

property_labels = ["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]

fig, axs = plt.subplots(encoding_dim, len(properties), figsize=(200, 500))

sns.set(font_scale=10)

for i, property in enumerate(properties):

    axs[0][i].set_title(property_labels[i])

    for feature in range(encoding_dim):

        axs[feature][i].scatter(x=extracted_features_switch[feature], y=all_properties[property])

        # sns.kdeplot(data=all_properties, x=extracted_features_switch[feature], y=all_properties[property], gridsize=200)

        axs[feature][i].set_xlabel("Feature " + str(feature), fontsize=75)
        axs[feature][i].set_ylabel(property_labels[i], fontsize=75)

plt.savefig("Correlation Plots Rand/scatter_" + str(encoding_dim) + "_feature_all_property_correlation_3_abs")
# plt.show()



# # sns.kdeplot(data=all_properties, x=extracted_features_switch[4], y=all_properties["MassType_Star"], levels=200, fill=True, cmap="mako")
#
# plt.hist2d(x=extracted_features_switch[4], y=all_properties["MassType_Star"], bins=50, cmap="jet")
#
# plt.show()

