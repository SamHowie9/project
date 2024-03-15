import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer


encoding_dim = 25


extracted_features = np.load("Features Rand/" + str(encoding_dim) + "_features_2.npy")


# load structural and physical properties into dataframes
structure_properties = pd.read_csv("Galaxy Properties/structure_propeties.csv", comment="#")
physical_properties = pd.read_csv("Galaxy Properties/physical_properties.csv", comment="#")

# account for hte validation data and remove final 200 elements
structure_properties.drop(structure_properties.tail(200).index, inplace=True)
physical_properties.drop(physical_properties.tail(200).index, inplace=True)

# dataframe for all properties
all_properties = pd.merge(structure_properties, physical_properties, on="GalaxyID")





# model = AgglomerativeClustering()
#
# visualizer = KElbowVisualizer(model, k=(1, 40), timings=False)
#
# visualizer.fit(extracted_features)        # Fit the data to the visualizer
# visualizer.show()






extracted_features_switch = np.flipud(np.rot90(extracted_features))

structure_correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude"])
physical_correlation_df = pd.DataFrame(columns=["Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])
correlation_df = pd.DataFrame(columns=["Sersic Index", "Position Angle", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"])

for feature in range(0, len(extracted_features_switch)):

    # create a list to contain the correlation between that feature and each property
    structure_correlation_list = []
    physical_correlation_list = []

    # loop through each property
    for gal_property in range(1, len(structure_properties.columns)):

        # calculate the correlation between that extracted feature and that property
        structure_correlation = np.corrcoef(extracted_features_switch[feature], structure_properties.iloc[:, gal_property])[0][1]
        structure_correlation_list.append(structure_correlation)

    for gal_property in range(1, len(physical_properties.columns)):

        # calculate the correlation between that extracted feature and that property
        physical_correlation = np.corrcoef(extracted_features_switch[feature], physical_properties.iloc[:, gal_property])[0][1]
        physical_correlation_list.append(physical_correlation)

    # add the correlation of that feature to the main dataframe
    structure_correlation_df.loc[len(structure_correlation_df)] = structure_correlation_list
    physical_correlation_df.loc[len(physical_correlation_df)] = physical_correlation_list
    correlation_df.loc[len(correlation_df)] = structure_correlation_list + physical_correlation_list


print(structure_correlation_df)
print(physical_correlation_df)
print(correlation_df)








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




# set the figure size
plt.figure(figsize=(20, 16))

sns.set(font_scale=1.5)

# plot a heatmap for the dataframe (with annotations)
ax = sns.heatmap(abs(correlation_df[["Semi - Major Axis", "Stellar Age", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation Coefficient'})


plt.yticks(rotation=0)
plt.ylabel("Extracted Features", fontsize=22)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(length=0)

ax.figure.axes[-1].yaxis.label.set_size(22)
ax.figure.axes[-1].yaxis.labelpad = 15


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0, fontsize=22)

wrap_labels(ax, 10)



plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_physical_property_correlation_2")
plt.show()






# # set the figure size
# plt.figure(figsize=(20, 16))
#
# # sns.set(font_scale = 2)
#
# # ["Sersic Index", "Axis Ratio", "Semi - Major Axis", "AB Magnitude", "Stellar Mass", "Gas Mass", "Dark Matter Mass", "Black Hole Particle Mass", "Black Hole Subgrid Mass", "Stellar Age", "Star Formation Rate"]
#
# # plot a heatmap for the dataframe (with annotations)
# # ax = sns.heatmap(abs(correlation_df[["Sersic Index", "Semi - Major Axis", "AB Magnitude", "Star Formation Rate", "Stellar Mass", "Dark Matter Mass", "Black Hole Mass"]]), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
# ax = sns.heatmap(abs(correlation_df), annot=True, cmap="Blues", cbar_kws={'label': 'Correlation'})
#
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
# plt.savefig("Correlation Plots Rand/" + str(encoding_dim) + "_feature_all_property_correlation_2")
# plt.show()
